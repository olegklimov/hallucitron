use anyhow::Result;
use futures_util::StreamExt;
use reqwest::Client;
use serde_json::{json, Value};

use super::openai_adapt;
use super::hallu_structs::{HalluStructuredRequest, HalluStructuredResult, HalluToolCall, HalluUsage};


const SSE_JUNK_KEYS: &[&str] = &[
    "item_id", "output_index", "logprobs", "content_index",
    "parallel_tool_calls", "previous_response_id", "reasoning",
    "tool_choice", "tools", "top_p", "top_logprobs",
    "presence_penalty", "frequency_penalty", "prompt_cache_key",
    "max_tool_calls", "safety_identifier", "store", "metadata",
    "background", "truncation", "user", "object",
    "annotations", "service_tier", "instructions",
    "incomplete_details", "text",
    "created_at", "completed_at", "id",
];

fn strip_sse_junk(v: &mut Value) {
    match v {
        Value::Object(obj) => {
            for key in SSE_JUNK_KEYS {
                obj.remove(*key);
            }
            obj.retain(|_, val| !val.is_null());
            for val in obj.values_mut() {
                strip_sse_junk(val);
            }
        }
        Value::Array(arr) => {
            for val in arr {
                strip_sse_junk(val);
            }
        }
        _ => {}
    }
}


pub async fn openai_streaming_call(req: &HalluStructuredRequest) -> Result<HalluStructuredResult> {
    let http = Client::new();
    let input = openai_adapt::adapt_messages(&req.messages, &http, &req.prov_api_key, &req.prov_endpoint, &req.prov_name).await?;

    let mut body = json!({
        "model": req.provm_name,
        "input": input,
        "stream": true,
    });
    apply_common_fields(req, &mut body);
    let mut log = vec![format!("openai_streaming: model={:?}", req.provm_name)];

    send_streaming_and_parse(req, &http, &body, &mut log).await
}


fn apply_common_fields(req: &HalluStructuredRequest, body: &mut Value) {
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
        // openai uses "reasoning" with effort; "max" is not supported, map to "high"
        let effort = if req.reasoning_effort == "max" { "high" } else { req.reasoning_effort.as_str() };
        body["reasoning"] = json!({"effort": effort});
    }
    if let Some(t) = req.temperature {
        if req.reasoning_effort.is_empty() || req.reasoning_effort == "none" {
            body["temperature"] = json!(t);
        }
    }
}


async fn send_streaming_and_parse(req: &HalluStructuredRequest, http: &Client, body: &Value, log: &mut Vec<String>) -> Result<HalluStructuredResult> {
    assert!(!req.prov_endpoint.is_empty(), "prov_endpoint must be set");
    let endpoint = format!("{}/responses", req.prov_endpoint.trim_end_matches('/'));

    super::dump_req_body(req, body);

    let response = http
        .post(&endpoint)
        .bearer_auth(&req.prov_api_key)
        .header("content-type", "application/json")
        .json(body)
        .send()
        .await?;

    if !response.status().is_success() {
        let status = response.status();
        let text = response.text().await.unwrap_or_default();
        anyhow::bail!("openai-compat streaming API error {status}: {text}");
    }

    let mut stream = response.bytes_stream();
    let mut buf = String::new();
    let mut text_pieces: Vec<String> = Vec::new();
    let mut resp_obj: Value = json!(null);
    let mut event_count: usize = 0;
    let mut pending_tool_calls: Vec<(String, String, Vec<String>)> = Vec::new();

    while let Some(chunk) = stream.next().await {
        let chunk = chunk?;
        buf.push_str(&String::from_utf8_lossy(&chunk));

        while let Some(pos) = buf.find("\n\n") {
            let block = buf[..pos].to_string();
            buf = buf[pos + 2..].to_string();

            for line in block.lines() {
                if let Some(data) = line.strip_prefix("data: ") {
                    event_count += 1;

                    if let Ok(mut ev) = serde_json::from_str::<Value>(data) {
                        let ev_type = ev["type"].as_str().unwrap_or("");
                        match ev_type {
                            "response.output_text.delta" => {
                                if let Some(d) = ev["delta"].as_str() {
                                    text_pieces.push(d.to_string());
                                    if let Some(tx) = &req.delta_tx {
                                        let _ = tx.try_send(d.to_string());
                                    }
                                }
                                strip_sse_junk(&mut ev);
                                log.push(format!("SSE #{event_count}: {ev}"));
                            }
                            "response.function_call_arguments.delta" => {
                                if let Some(d) = ev["delta"].as_str() {
                                    if let Some(last) = pending_tool_calls.last_mut() {
                                        last.2.push(d.to_string());
                                    }
                                }
                                strip_sse_junk(&mut ev);
                                log.push(format!("SSE #{event_count}: {ev}"));
                            }
                            "response.output_item.added" => {
                                let item = &ev["item"];
                                if item["type"].as_str() == Some("function_call") {
                                    let call_id = item["call_id"].as_str().unwrap_or("").to_string();
                                    let name = item["name"].as_str().unwrap_or("").to_string();
                                    pending_tool_calls.push((call_id, name, Vec::new()));
                                }
                                strip_sse_junk(&mut ev);
                                log.push(format!("SSE #{event_count}: {ev}"));
                            }
                            "response.completed" => {
                                resp_obj = ev["response"].clone();
                                strip_sse_junk(&mut ev);
                                log.push(format!("SSE #{event_count}: {ev}"));
                            }
                            _ => {
                                strip_sse_junk(&mut ev);
                                log.push(format!("SSE #{event_count}: {ev}"));
                            }
                        }
                    }
                }
            }
        }
    }

    let raw_text: String = text_pieces.concat();

    let tool_calls: Vec<HalluToolCall> = pending_tool_calls.into_iter().map(|(call_id, name, arg_pieces)| {
        let args_str = arg_pieces.concat();
        let arguments: Value = serde_json::from_str(&args_str).unwrap_or(json!({}));
        HalluToolCall { call_id, name, arguments }
    }).collect();

    if raw_text.trim().is_empty() && tool_calls.is_empty() {
        anyhow::bail!("openai_streaming: no text output and no tool calls after {event_count} events, max_tokens={}, resp={resp_obj}\n{}", req.max_tokens, log.join("\n"));
    }

    let parsed: Value = if !tool_calls.is_empty() {
        Value::Null
    } else if req.output_schema.is_null() {
        Value::String(raw_text.clone())
    } else {
        serde_json::from_str(&raw_text)
            .map_err(|e| anyhow::anyhow!("failed to parse structured JSON: {e}\nraw: {raw_text}"))?
    };

    let actual_model = resp_obj["model"].as_str().unwrap_or(&req.provm_name).to_string();
    let stop_reason = resp_obj["status"].as_str().unwrap_or("").to_string();
    let response_id = resp_obj["id"].as_str().unwrap_or("").to_string();

    let usage = &resp_obj["usage"];
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

    log.push(format!("usage: {}", usage));

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
        stop_reason,
        response_id,
        provider_usage_json,
        sse_log: std::mem::take(log),
    })
}
