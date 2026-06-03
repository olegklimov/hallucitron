use anyhow::{anyhow, Result};
use reqwest::Client;
use serde_json::{json, Value};

use super::anthropic_adapt;
use super::hallu_structs::{HalluStructuredRequest, HalluStructuredResult, HalluToolCall, HalluUsage};


const ANTHROPIC_VERSION: &str = "2023-06-01";

// No thinking stream available for athropic models.


pub async fn anthropic_structured_call(req: &HalluStructuredRequest) -> Result<HalluStructuredResult> {
    let http = Client::new();
    let adapted = anthropic_adapt::adapt_messages(&req.messages, &http, &req.prov_api_key, &req.prov_endpoint).await?;

    assert!(!req.prov_endpoint.is_empty(), "prov_endpoint must be set");
    let endpoint = format!("{}/messages", req.prov_endpoint.trim_end_matches('/'));

    let mut body = json!({
        "model": req.provm_name,
        "max_tokens": req.max_tokens,
        "system": adapted.system,
        "messages": adapted.messages,
        "cache_control": {"type": "ephemeral", "ttl": "1h"},
    });
    if !req.output_schema.is_null() {
        assert!(
            req.output_schema.get("additionalProperties") == Some(&json!(false)),
            "output_schema must have additionalProperties: false (anthropic requirement)"
        );
        body["output_config"] = json!({
            "format": {
                "type": "json_schema",
                "schema": req.output_schema,
            }
        });
    }
    if let Some(tools) = req.tools.as_array() {
        if !tools.is_empty() {
            // convert openai-style tools to anthropic format
            let anthropic_tools: Vec<Value> = tools.iter().filter_map(|t| {
                if t["type"].as_str() == Some("function") {
                    Some(json!({
                        "name": t["name"],
                        "description": t["description"],
                        "input_schema": t["parameters"],
                    }))
                } else {
                    Some(t.clone())
                }
            }).collect();
            body["tools"] = json!(anthropic_tools);
            if !req.output_schema.is_null() {
                body["tool_choice"] = json!({"type": "none"});  // prevent tool calls (removing tools instead would break cache)
            }
        }
    }
    let thinking_on = super::anthropic_streaming::apply_anthropic_thinking(req, &mut body);
    if let Some(t) = req.temperature {
        if !thinking_on {
            body["temperature"] = json!(t);
        }
    }

    super::dump_req_body(req, &body);

    let mut rb = http
        .post(&endpoint)
        .header("x-api-key", &req.prov_api_key)
        .header("anthropic-version", ANTHROPIC_VERSION)
        .header("content-type", "application/json");
    if adapted.needs_files_beta {
        rb = rb.header("anthropic-beta", anthropic_adapt::ANTHROPIC_FILES_BETA);
    }
    let response = rb.json(&body).send().await?;

    if !response.status().is_success() {
        let status = response.status();
        let text = response.text().await.unwrap_or_default();
        anyhow::bail!("anthropic API error {status}: {text}");
    }

    let resp: Value = response.json().await?;
    parse_anthropic_response(&resp, req)
}


fn parse_anthropic_response(resp: &Value, req: &HalluStructuredRequest) -> Result<HalluStructuredResult> {
    let actual_model = resp["model"].as_str().unwrap_or(&req.provm_name).to_string();
    let stop_reason = resp["stop_reason"].as_str().unwrap_or("").to_string();
    let response_id = resp["id"].as_str().unwrap_or("").to_string();

    let mut raw_text = String::new();
    let mut thinking_text = String::new();
    let mut thinking_blocks: Vec<Value> = Vec::new();
    let mut tool_calls: Vec<HalluToolCall> = Vec::new();
    if let Some(content) = resp["content"].as_array() {
        for block in content {
            match block["type"].as_str() {
                Some("text") => {
                    raw_text.push_str(block["text"].as_str().unwrap_or(""));
                }
                Some("thinking") => {
                    if !thinking_text.is_empty() { thinking_text.push('\n'); }
                    thinking_text.push_str(block["thinking"].as_str().unwrap_or(""));
                    thinking_blocks.push(block.clone());
                }
                Some("tool_use") => {
                    tool_calls.push(HalluToolCall {
                        call_id: block["id"].as_str().unwrap_or("").to_string(),
                        name: block["name"].as_str().unwrap_or("").to_string(),
                        arguments: block["input"].clone(),
                    });
                }
                _ => {}
            }
        }
    }

    if raw_text.trim().is_empty() && tool_calls.is_empty() {
        anyhow::bail!(
            "anthropic_call: no text output and no tool calls, max_tokens={}, stop_reason={stop_reason:?}, response={resp}", req.max_tokens,
        );
    }

    let parsed: Value = if !tool_calls.is_empty() {
        if raw_text.trim().is_empty() {
            Value::Null
        } else {
            Value::String(raw_text.clone())
        }
    } else if req.output_schema.is_null() {
        Value::String(raw_text.clone())
    } else {
        serde_json::from_str(&raw_text)
            .map_err(|e| anyhow!("failed to parse structured JSON from response: {e}\nraw: {raw_text}"))?
    };

    let usage = &resp["usage"];
    let provider_usage_json = usage.clone();
    let input_tokens = usage["input_tokens"].as_u64().unwrap_or(0);
    let output_tokens = usage["output_tokens"].as_u64().unwrap_or(0);
    let cache_creation = usage["cache_creation_input_tokens"].as_u64().unwrap_or(0);
    let cache_read = usage["cache_read_input_tokens"].as_u64().unwrap_or(0);

    let provider_specific_stuff = if thinking_blocks.is_empty() {
        Value::Null
    } else {
        json!(thinking_blocks)
    };

    Ok(HalluStructuredResult {
        parsed,
        raw_text,
        thinking_text,
        provider_specific_stuff,
        tool_calls,
        usage: HalluUsage {
            prompt_noncached: input_tokens,
            output_tokens,
            cache_creation_input_tokens: cache_creation,
            cache_read_input_tokens: cache_read,
            ..Default::default()
        },
        coins: 0,
        price_breakdown: Vec::new(),
        provider_cost_usd: None,
        actual_model,
        stop_reason,
        response_id,
        provider_usage_json,
        sse_log: Vec::new(),
    })
}


