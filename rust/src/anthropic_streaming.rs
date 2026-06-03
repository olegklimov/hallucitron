use anyhow::{anyhow, Result};
use futures_util::StreamExt;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::BTreeMap;

use super::anthropic_adapt;
use super::hallu_structs::{HalluStructuredRequest, HalluStructuredResult, HalluToolCall, HalluUsage};


const ANTHROPIC_VERSION: &str = "2023-06-01";


#[derive(Debug, Default, Clone, Serialize, Deserialize)]
struct WireUsage {
    input_tokens: Option<u64>,
    output_tokens: Option<u64>,
    cache_creation_input_tokens: Option<u64>,
    cache_read_input_tokens: Option<u64>,
}


#[derive(Debug, Clone)]
enum BlockState {
    Text { text: String },
    ToolUse { id: String, name: String, partial_json: String },
    Thinking { thinking: String, signature: String },
}


const SSE_JUNK_KEYS: &[&str] = &[
    "content_block", "index", "logprobs",
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


// Translates reasoning_effort into anthropic-specific thinking + effort params.
// Returns true if thinking was enabled (temperature must be omitted in that case).
// effort controls thinking depth, budget_tokens is just the ceiling.
pub fn apply_anthropic_thinking(req: &HalluStructuredRequest, body: &mut Value) -> bool {
    if req.reasoning_effort.is_empty() {
        return false;
    }
    let effort = req.reasoning_effort.as_str();
    if req.provm_name == "claude-opus-4-6" || req.provm_name == "claude-sonnet-4-6" {
        body["thinking"] = json!({"type": "adaptive"});
    } else {
        // budget_tokens must be >= 1024 (anthropic minimum) and < max_tokens
        if req.max_tokens < 2048 {
            tracing::warn!("apply_anthropic_thinking: max_tokens={} too low for thinking (budget_tokens must be >= 1024), skipping", req.max_tokens);
            return false;
        }
        let budget = (req.max_tokens / 2).max(1024);
        body["thinking"] = json!({"type": "enabled", "budget_tokens": budget});
    }
    if effort != "high" {
        if !body.get("output_config").map(Value::is_object).unwrap_or(false) {
            body["output_config"] = json!({});
        }
        body["output_config"]["effort"] = json!(effort);
    }
    true
}


pub async fn anthropic_streaming_call(req: &HalluStructuredRequest) -> Result<HalluStructuredResult> {
    let http = Client::new();
    let adapted = anthropic_adapt::adapt_messages(&req.messages, &http, &req.prov_api_key, &req.prov_endpoint).await?;

    assert!(!req.prov_endpoint.is_empty(), "prov_endpoint must be set");
    let endpoint = format!("{}/messages", req.prov_endpoint.trim_end_matches('/'));

    let mut body = json!({
        "model": req.provm_name,
        "max_tokens": req.max_tokens,
        "system": adapted.system,
        "messages": adapted.messages,
        "stream": true,
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
    let thinking_on = apply_anthropic_thinking(req, &mut body);
    if let Some(t) = req.temperature {
        // anthropic doesn't allow temperature with thinking enabled
        if !thinking_on {
            body["temperature"] = json!(t);
        }
    }

    super::dump_req_body(req, &body);

    let mut log = vec![format!("anthropic_streaming: model={:?}", req.provm_name)];

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
        anyhow::bail!("anthropic streaming API error {status}: {text}");
    }

    let mut stream = response.bytes_stream();
    let mut buf = String::new();
    let mut blocks: BTreeMap<usize, BlockState> = BTreeMap::new();
    let mut usage = WireUsage::default();
    let mut stop_reason: Option<String> = None;
    let mut actual_model: Option<String> = None;
    let mut response_id: Option<String> = None;
    let mut provider_usage_json: Value = json!(null);
    let mut event_count: usize = 0;

    while let Some(chunk) = stream.next().await {
        let chunk = chunk?;
        buf.push_str(&String::from_utf8_lossy(&chunk).replace("\r\n", "\n"));

        while let Some(pos) = buf.find("\n\n") {
            let frame = buf[..pos].to_string();
            buf.drain(..pos + 2);

            let mut event_type = String::new();
            let mut data_lines = Vec::new();
            for line in frame.lines() {
                if let Some(v) = line.strip_prefix("event:") {
                    event_type = v.trim().to_string();
                } else if let Some(v) = line.strip_prefix("data:") {
                    data_lines.push(v.trim_start().to_string());
                }
            }
            if data_lines.is_empty() { continue; }
            let data = data_lines.join("\n");
            if data == "[DONE]" { break; }

            let mut v: Value = match serde_json::from_str(&data) {
                Ok(v) => v,
                Err(_) => continue,
            };
            event_count += 1;

            // model + response id from message_start
            if event_type == "message_start" {
                if let Some(m) = v.pointer("/message/model").and_then(Value::as_str) {
                    actual_model = Some(m.to_string());
                }
                if let Some(id) = v.pointer("/message/id").and_then(Value::as_str) {
                    response_id = Some(id.to_string());
                }
            }

            // usage: input from message_start.message.usage, output from message_delta.usage
            if let Some(u) = v.pointer("/message/usage").or_else(|| v.get("usage")) {
                provider_usage_json = u.clone();
                if let Ok(parsed) = serde_json::from_value::<WireUsage>(u.clone()) {
                    if parsed.input_tokens.unwrap_or(0) > 0 {
                        usage.input_tokens = parsed.input_tokens;
                    }
                    if parsed.output_tokens.unwrap_or(0) > 0 {
                        usage.output_tokens = parsed.output_tokens;
                    }
                    if parsed.cache_creation_input_tokens.unwrap_or(0) > 0 {
                        usage.cache_creation_input_tokens = parsed.cache_creation_input_tokens;
                    }
                    if parsed.cache_read_input_tokens.unwrap_or(0) > 0 {
                        usage.cache_read_input_tokens = parsed.cache_read_input_tokens;
                    }
                }
            }

            if let Some(sr) = v.pointer("/delta/stop_reason").and_then(Value::as_str) {
                stop_reason = Some(sr.to_string());
            }

            match event_type.as_str() {
                "content_block_start" => {
                    let index = v["index"].as_u64().unwrap_or(0) as usize;
                    let block = &v["content_block"];
                    match block["type"].as_str() {
                        Some("text") => {
                            blocks.insert(index, BlockState::Text { text: String::new() });
                        }
                        Some("tool_use") => {
                            blocks.insert(index, BlockState::ToolUse {
                                id: block["id"].as_str().unwrap_or("").to_string(),
                                name: block["name"].as_str().unwrap_or("").to_string(),
                                partial_json: String::new(),
                            });
                        }
                        Some("thinking") => {
                            blocks.insert(index, BlockState::Thinking {
                                thinking: String::new(),
                                signature: String::new(),
                            });
                        }
                        _ => {}
                    }
                    strip_sse_junk(&mut v);
                    log.push(format!("SSE #{event_count}: {v}"));
                }
                "content_block_delta" => {
                    let index = v["index"].as_u64().unwrap_or(0) as usize;
                    let delta = &v["delta"];
                    match delta["type"].as_str() {
                        Some("text_delta") => {
                            let chunk = delta["text"].as_str().unwrap_or("");
                            if let Some(BlockState::Text { text }) = blocks.get_mut(&index) {
                                text.push_str(chunk);
                                if let Some(tx) = &req.delta_tx {
                                    let _ = tx.try_send(chunk.to_string());
                                }
                            }
                        }
                        Some("input_json_delta") => {
                            let chunk = delta["partial_json"].as_str().unwrap_or("");
                            if let Some(BlockState::ToolUse { partial_json, .. }) = blocks.get_mut(&index) {
                                partial_json.push_str(chunk);
                            }
                        }
                        Some("thinking_delta") => {
                            let chunk = delta["thinking"].as_str().unwrap_or("");
                            if let Some(BlockState::Thinking { thinking, .. }) = blocks.get_mut(&index) {
                                thinking.push_str(chunk);
                            }
                        }
                        Some("signature_delta") => {
                            let chunk = delta["signature"].as_str().unwrap_or("");
                            if let Some(BlockState::Thinking { signature, .. }) = blocks.get_mut(&index) {
                                signature.push_str(chunk);
                            }
                        }
                        _ => {}
                    }
                    strip_sse_junk(&mut v);
                    log.push(format!("SSE #{event_count}: {v}"));
                }
                "content_block_stop" | "message_start" | "message_delta" | "message_stop" => {
                    strip_sse_junk(&mut v);
                    log.push(format!("SSE #{event_count}: {v}"));
                }
                "ping" => {}
                _ => {
                    strip_sse_junk(&mut v);
                    log.push(format!("SSE #{event_count}: {v}"));
                }
            }
        }
    }

    // Assemble results from blocks
    let mut raw_text = String::new();
    let mut thinking_text = String::new();
    let mut thinking_blocks: Vec<Value> = Vec::new();
    let mut tool_calls: Vec<HalluToolCall> = Vec::new();
    for (_, block) in blocks {
        match block {
            BlockState::Text { text } => {
                raw_text.push_str(&text);
            }
            BlockState::Thinking { thinking, signature } => {
                if !thinking_text.is_empty() { thinking_text.push('\n'); }
                thinking_text.push_str(&thinking);
                thinking_blocks.push(json!({"type": "thinking", "thinking": thinking, "signature": signature}));
            }
            BlockState::ToolUse { id, name, partial_json } => {
                let arguments: Value = if partial_json.trim().is_empty() {
                    json!({})
                } else {
                    serde_json::from_str(&partial_json).unwrap_or(json!({}))
                };
                tool_calls.push(HalluToolCall { call_id: id, name, arguments });
            }
        }
    }

    if raw_text.trim().is_empty() && tool_calls.is_empty() {
        anyhow::bail!(
            "anthropic_streaming: no text output and no tool calls after {event_count} events, max_tokens={}, stop_reason={stop_reason:?}\n{}", req.max_tokens, log.join("\n"),
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
            .map_err(|e| anyhow!("failed to parse structured JSON: {e}\nraw: {raw_text}"))?
    };

    let input_tokens = usage.input_tokens.unwrap_or(0);
    let output_tokens = usage.output_tokens.unwrap_or(0);
    let cache_creation = usage.cache_creation_input_tokens.unwrap_or(0);
    let cache_read = usage.cache_read_input_tokens.unwrap_or(0);

    log.push(format!("usage: input={input_tokens} output={output_tokens} cache_creation={cache_creation} cache_read={cache_read}"));

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
        provider_usage_json,
        actual_model: actual_model.unwrap_or_else(|| req.provm_name.clone()),
        stop_reason: stop_reason.unwrap_or_default(),
        response_id: response_id.unwrap_or_default(),
        sse_log: log,
    })
}
