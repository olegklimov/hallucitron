use anyhow::{anyhow, Result};
use reqwest::Client;
use serde_json::{json, Value};

use super::openai_adapt;
use super::hallu_structs::{HalluStructuredRequest, HalluStructuredResult, HalluToolCall, HalluUsage};

// "thinking tokens streaming": you can’t stream raw thinking tokens. OpenAI’s docs say raw reasoning tokens are not visible via the API.
// The closest thing you can enable is a reasoning summary with reasoning.summary, and that summary comes back in a reasoning output item.
// struct OutputItem {
//   ...
//     summary: Option<Vec<ContentPart>>, // add this
// }

// include: ["reasoning.encrypted_content"] is for carrying reasoning across turns in stateless mode


pub async fn openai_structured_call(req: &HalluStructuredRequest) -> Result<HalluStructuredResult> {
    let http = Client::new();
    let input = openai_adapt::adapt_messages(&req.messages, &http, &req.prov_api_key, &req.prov_endpoint, &req.prov_name).await?;

    assert!(!req.prov_endpoint.is_empty(), "prov_endpoint must be set");
    let endpoint = format!("{}/responses", req.prov_endpoint.trim_end_matches('/'));

    let mut body = json!({
        "model": req.provm_name,
        "input": input,
    });
    if !req.output_schema.is_null() {
        body["text"] = json!({
            "format": {
                "type": "json_schema",
                "name": req.output_schema_name,
                "schema": req.output_schema,
                "strict": true,
            }
        });
    }
    if let Some(tools) = req.tools.as_array() {
        if !tools.is_empty() {
            body["tools"] = req.tools.clone();
            if !req.output_schema.is_null() {
                body["tool_choice"] = json!("none");  // prevent tool calls (removing tools instead would break cache)
            }
        }
    }
    if req.max_tokens > 0 {
        body["max_output_tokens"] = json!(req.max_tokens);
    }
    if !req.reasoning_effort.is_empty() {
        let effort = if req.reasoning_effort == "max" { "high" } else { req.reasoning_effort.as_str() };
        body["reasoning"] = json!({"effort": effort});
    }
    if let Some(t) = req.temperature {
        if req.reasoning_effort.is_empty() || req.reasoning_effort == "none" {
            body["temperature"] = json!(t);
        }
    }

    super::dump_req_body(req, &body);

    let response = http
        .post(&endpoint)
        .bearer_auth(&req.prov_api_key)
        .header("content-type", "application/json")
        .json(&body)
        .send()
        .await?;

    if !response.status().is_success() {
        let status = response.status();
        let text = response.text().await.unwrap_or_default();
        anyhow::bail!("openai-compat API error {status}: {text}");
    }

    let resp: Value = response.json().await?;
    let actual_model = resp["model"].as_str().unwrap_or(&req.provm_name).to_string();
    let response_id = resp["id"].as_str().unwrap_or("").to_string();

    let mut raw_text = String::new();
    let mut tool_calls: Vec<HalluToolCall> = Vec::new();
    if let Some(output) = resp["output"].as_array() {
        for item in output {
            match item["type"].as_str() {
                Some("message") => {
                    if let Some(content) = item["content"].as_array() {
                        for part in content {
                            if part["type"].as_str() == Some("output_text") {
                                raw_text.push_str(part["text"].as_str().unwrap_or(""));
                            }
                        }
                    }
                }
                Some("function_call") => {
                    let name = item["name"].as_str().unwrap_or("").to_string();
                    let call_id = item["call_id"].as_str().unwrap_or("").to_string();
                    let args_str = item["arguments"].as_str().unwrap_or("{}");
                    let arguments: Value = serde_json::from_str(args_str).unwrap_or(json!({}));
                    tool_calls.push(HalluToolCall { call_id, name, arguments });
                }
                _ => {}
            }
        }
    }

    if raw_text.trim().is_empty() && tool_calls.is_empty() {
        anyhow::bail!("openai_responses_api: no text output and no tool calls, max_tokens={}, response={resp}", req.max_tokens);
    }

    let parsed: Value = if !tool_calls.is_empty() {
        Value::Null
    } else if req.output_schema.is_null() {
        Value::String(raw_text.clone())
    } else {
        serde_json::from_str(&raw_text)
            .map_err(|e| anyhow!("failed to parse structured JSON: {e}\nraw: {raw_text}"))?
    };

    let usage = &resp["usage"];
    let provider_usage_json = usage.clone();
    let input_tokens = usage["input_tokens"].as_u64().unwrap_or(0);
    let cached_tokens = usage.pointer("/input_tokens_details/cached_tokens")
        .and_then(Value::as_u64).unwrap_or(0);
    let reasoning_tokens = usage.pointer("/output_tokens_details/reasoning_tokens")
        .and_then(Value::as_u64).unwrap_or(0);
    let provider_cost_usd = usage["cost_in_usd_ticks"].as_f64().map(|t| t / 10_000_000_000.0);
    let tool_detail = |name: &str| -> u64 {
        usage.pointer(&format!("/server_side_tool_usage_details/{name}"))
            .and_then(Value::as_u64).unwrap_or(0)
    };
    Ok(HalluStructuredResult {
        parsed,
        raw_text,
        thinking_text: String::new(),
        provider_specific_stuff: Value::Null,
        tool_calls,
        usage: HalluUsage {
            prompt_noncached: input_tokens.saturating_sub(cached_tokens),
            output_tokens: usage["output_tokens"].as_u64().unwrap_or(0),
            including_reasoning_tokens: reasoning_tokens,
            cache_creation_input_tokens: 0,
            cache_read_input_tokens: cached_tokens,
            call_web_search: tool_detail("web_search_calls"),
            call_x_search: tool_detail("x_search_calls"),
            call_code_interpreter: tool_detail("code_interpreter_calls"),
            call_document_search: tool_detail("document_search_calls"),
            call_file_search: tool_detail("file_search_calls"),
            ..Default::default()
        },
        coins: 0,
        price_breakdown: Vec::new(),
        provider_cost_usd,
        actual_model,
        stop_reason: resp["status"].as_str().unwrap_or("").to_string(),
        response_id,
        provider_usage_json,
        sse_log: Vec::new(),
    })
}
