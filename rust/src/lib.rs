use serde_json::Value;

use hallu_structs::{HalluStructuredRequest, HalluStructuredResult};

pub mod hallu_providers;
#[allow(dead_code)] pub mod hallu_structs;
#[allow(dead_code)] pub mod anthropic_adapt;
#[allow(dead_code)] pub mod anthropic_non_streaming;
#[allow(dead_code)] pub mod anthropic_streaming;
#[allow(dead_code)] pub mod openai_adapt;
#[allow(dead_code)] pub mod openai_non_streaming;
#[allow(dead_code)] pub mod openai_streaming;

#[cfg(test)] mod test_data;
#[cfg(test)] mod test_anthropic;
#[cfg(test)] mod test_openai;
#[cfg(test)] mod test_xai;


pub fn dump_req_body(req: &HalluStructuredRequest, body: &Value) {
    if req.req_dump_path.is_empty() { return; }
    let path = std::path::Path::new(&req.req_dump_path);
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).ok();
    }
    let pretty = serde_json::to_string_pretty(body).unwrap_or_default();
    if let Err(e) = std::fs::write(path, &pretty) {
        tracing::warn!("dump_req_body failed {}: {e}", path.display());
    }
}

pub async fn hallu_call(req: &HalluStructuredRequest) -> anyhow::Result<HalluStructuredResult> {
    let model = &req.provm_name;
    let is_openai_compat = model.starts_with("gpt-") || model.starts_with("o1-") || model.starts_with("o3-") || model.starts_with("o4-") || model.starts_with("grok-");
    let mut result = if model.starts_with("claude-") && req.streaming {
        anthropic_streaming::anthropic_streaming_call(req).await?
    } else if model.starts_with("claude-") {
        anthropic_non_streaming::anthropic_structured_call(req).await?
    } else if is_openai_compat && req.streaming {
        openai_streaming::openai_streaming_call(req).await?
    } else if is_openai_compat {
        openai_non_streaming::openai_structured_call(req).await?
    } else {
        anyhow::bail!("hallu_call: unknown model prefix for {model:?}")
    };
    result.usage.input_images = count_input_images(&req.messages);
    apply_prices(&mut result, &req.provm_prices);
    Ok(result)
}


fn count_input_images(messages: &[hallu_structs::HalluMessage]) -> u64 {
    let mut n: u64 = 0;
    for msg in messages {
        if let Some(parts) = msg.content.as_array() {
            for part in parts {
                let t = part["m_type"].as_str().unwrap_or("");
                if t.starts_with("image/") {
                    n += 1;
                }
            }
        }
    }
    n
}

fn apply_prices(result: &mut HalluStructuredResult, prices: &Value) {
    let (coins, breakdown) = convolute_usage_with_prices(&result.usage, prices);
    result.coins = coins;
    result.price_breakdown = breakdown;
}


// Match non-zero usage fields against pp1000t_* prices, warn on missing prices.
// input_tokens meaning differs: anthropic excludes cache_read, openai includes it.
// So prompt_noncached is computed here to work for both.
fn convolute_usage_with_prices(usage: &hallu_structs::HalluUsage, prices: &Value) -> (i64, Vec<String>) {
    let mut breakdown = Vec::new();
    let pairs: &[(&str, u64)] = &[
        ("pp1000t_prompt", usage.prompt_noncached),
        ("pp1000t_prompt_text", usage.prompt_noncached),   // pp1000t_prompt_text is non zero when pp1000t_prompt is zero (grok modality breakdown)
        ("pp1000t_prompt_without_cache", usage.prompt_noncached),
        ("pp1000t_prompt_cached", usage.cache_read_input_tokens),
        ("pp1000t_cache_read", usage.cache_read_input_tokens),
        ("pp1000t_cache_creation", usage.cache_creation_input_tokens),
        ("pp1000t_completion", usage.output_tokens),
    ];
    let mut coins: i64 = 0;
    for &(key, tokens) in pairs {
        if tokens == 0 { continue; }
        if let Some(p) = prices.get(key).and_then(Value::as_i64) {
            let c = (tokens as i64 * p) / 1000;
            let line = format!("{} = {} * {} / 1000 {}", c, p, tokens, key);
            breakdown.push(line);
            coins += c;
        }
    }
    // per-unit billing: server-side tool calls (pp1call_*) and per-image (pp1image)
    let tool_pairs: &[(&str, u64)] = &[
        ("pp1call_web_search", usage.call_web_search),
        ("pp1call_x_search", usage.call_x_search),
        ("pp1call_code_interpreter", usage.call_code_interpreter),
        ("pp1call_document_search", usage.call_document_search),
        ("pp1call_file_search", usage.call_file_search),
        ("pp1image", usage.input_images),
    ];
    for &(key, calls) in tool_pairs {
        if calls == 0 { continue; }
        if let Some(p) = prices.get(key).and_then(Value::as_i64) {
            let c = calls as i64 * p;
            let line = format!("{} = {} * {} {}", c, calls, p, key);
            breakdown.push(line);
            coins += c;
        }
    }
    (coins, breakdown)
}
