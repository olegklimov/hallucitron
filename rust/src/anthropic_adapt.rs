use std::collections::HashMap;
use std::sync::Mutex;
use std::time::Instant;

use anyhow::Result;
use reqwest::{multipart, Client};
use serde::Deserialize;
use serde_json::{json, Value};

use super::hallu_structs::HalluMessage;


const ANTHROPIC_VERSION: &str = "2023-06-01";
pub const ANTHROPIC_FILES_BETA: &str = "files-api-2025-04-14";

static FILE_CACHE: std::sync::LazyLock<Mutex<HashMap<String, (String, Instant)>>> =
    std::sync::LazyLock::new(|| Mutex::new(HashMap::new()));

const CACHE_TTL_SECS: u64 = 24 * 3600;

#[derive(Deserialize)]
struct UploadedFile {
    id: String,
}


// Converts flexus HalluMessage list into Anthropic messages API format.
// Returns (system_content, messages) where system is extracted separately
// because Anthropic wants it as a top-level field.

pub struct AnthropicAdapted {
    pub system: Vec<Value>,      // anthropic system blocks [{type: "text", text: "...", cache_control?}]
    pub messages: Vec<Value>,    // anthropic messages [{role, content}]
    pub needs_files_beta: bool,  // true if any file uploads were used
}


pub async fn adapt_messages(msgs: &[HalluMessage], http: &Client, api_key: &str, api_endpoint: &str) -> Result<AnthropicAdapted> {
    let mut system_blocks: Vec<Value> = Vec::new();
    let mut messages: Vec<Value> = Vec::new();
    let mut needs_files_beta = false;

    for m in msgs {
        match m.role.as_str() {
            "system" => {
                let (blocks, used_files) = content_to_anthropic_blocks(&m.content, http, api_key, api_endpoint).await?;
                needs_files_beta |= used_files;
                system_blocks.extend(blocks);
            }
            "assistant" => {
                // provider_specific_stuff carries anthropic-native blocks (thinking+signature) that must go first
                let mut blocks: Vec<Value> = Vec::new();
                if let Some(pss) = m.provider_specific_stuff.as_array() {
                    blocks.extend(pss.iter().cloned());
                }
                let (content_blocks, used_files) = content_to_anthropic_blocks(&m.content, http, api_key, api_endpoint).await?;
                needs_files_beta |= used_files;
                blocks.extend(content_blocks);
                if let Some(tool_calls) = m.tool_calls.as_array() {
                    for tc in tool_calls {
                        let f = &tc["function"];
                        let input: Value = serde_json::from_str(
                            f["arguments"].as_str().unwrap_or("{}")
                        ).unwrap_or(json!({}));
                        blocks.push(json!({
                            "type": "tool_use",
                            "id": tc["id"].as_str().unwrap_or(""),
                            "name": f["name"].as_str().unwrap_or(""),
                            "input": input,
                        }));
                    }
                }
                messages.push(json!({"role": "assistant", "content": blocks}));
            }
            "user" | "context_file" | "hint" | "plain_text" => {
                let (blocks, used_files) = content_to_anthropic_blocks(&m.content, http, api_key, api_endpoint).await?;
                assert!(!blocks.is_empty(), "what's that (5)\n{:?}", m.content);
                needs_files_beta |= used_files;
                messages.push(json!({"role": "user", "content": blocks}));
            }
            "cd_instruction" => {
                let (mut blocks, used_files) = content_to_anthropic_blocks(&m.content, http, api_key, api_endpoint).await?;
                assert!(!blocks.is_empty(), "what's that (6)\n{:?}", m.content);
                assert_eq!(blocks[0]["type"], "text", "cd_instruction first block not text: {:?}", blocks[0]);
                let text = blocks[0]["text"].as_str().expect("cd_instruction first block text not str");
                blocks[0]["text"] = json!(format!("\u{1f4bf} {}", text));
                needs_files_beta |= used_files;
                messages.push(json!({"role": "user", "content": blocks}));
            }
            "tool" | "diff" => {
                let text = content_to_text(&m.content);
                messages.push(json!({
                    "role": "user",
                    "content": [{
                        "type": "tool_result",
                        "tool_use_id": m.call_id,
                        "content": text,
                    }]
                }));
            }
            "title" | "cork" => {}
            other => {
                tracing::warn!("anthropic_adapt: unknown role={other:?}, skipping");
            }
        }
    }

    Ok(AnthropicAdapted { system: system_blocks, messages, needs_files_beta })
}


async fn content_to_anthropic_blocks(content: &Value, http: &Client, api_key: &str, api_endpoint: &str) -> Result<(Vec<Value>, bool)> {
    let mut used_files = false;
    match content {
        Value::Null => Ok((vec![], false)),
        Value::String(s) => {
            if s.is_empty() {
                Ok((vec![], false))
            } else {
                Ok((vec![json!({"type": "text", "text": s})], false))
            }
        }
        Value::Array(parts) => {
            let mut blocks = Vec::new();
            for part in parts {
                let m_type = part["m_type"].as_str().unwrap_or("");
                let m_content = part["m_content"].as_str().unwrap_or("");
                match m_type {
                    "text" => {
                        if !m_content.trim().is_empty() {
                            blocks.push(json!({"type": "text", "text": m_content}));
                        }
                    }
                    t if t.starts_with("image/") => {
                        if m_content.starts_with("http://") || m_content.starts_with("https://") || m_content.starts_with("file://") {
                            let file_id = upload_or_cached(http, api_key, api_endpoint, m_content, t).await?;
                            used_files = true;
                            blocks.push(json!({
                                "type": "image",
                                "source": {
                                    "type": "file",
                                    "file_id": file_id,
                                }
                            }));
                        } else {
                            blocks.push(json!({
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": t,
                                    "data": m_content,
                                }
                            }));
                        }
                    }
                    "pdf" => {
                        let file_id = upload_or_cached(http, api_key, api_endpoint, m_content, "application/pdf").await?;
                        used_files = true;
                        blocks.push(json!({
                            "type": "document",
                            "source": {
                                "type": "file",
                                "file_id": file_id,
                            }
                        }));
                    }
                    _ => {
                        blocks.push(json!({"type": "text", "text": m_content}));
                    }
                }
            }
            Ok((blocks, used_files))
        }
        other => {
            tracing::error!("something horrible: {:?}", other);
            Ok((vec![json!({"type": "text", "text": "unknown stuff :|"})], false))
        }
    }
}


fn content_to_text(content: &Value) -> String {
    match content {
        Value::Null => String::new(),
        Value::String(s) => s.clone(),
        Value::Array(parts) => {
            parts.iter().map(|p| {
                p["m_content"].as_str().unwrap_or("").to_string()
            }).collect::<Vec<_>>().join(" ")
        }
        other => other.to_string(),
    }
}



async fn upload_or_cached(http: &Client, api_key: &str, api_endpoint: &str, url: &str, content_type: &str) -> Result<String> {
    let cache_key = format!("{}:{}", api_endpoint, url);

    {
        let cache = FILE_CACHE.lock().unwrap();
        if let Some((file_id, ts)) = cache.get(&cache_key) {
            if ts.elapsed().as_secs() < CACHE_TTL_SECS {
                tracing::debug!("anthropic file cache hit: {url} -> {file_id}");
                return Ok(file_id.clone());
            }
        }
    }

    let bytes = download_raw(http, url).await?;
    let filename = url.rsplit('/').next().unwrap_or("document.pdf").to_string();

    // derive files endpoint from api_endpoint (e.g. https://api.anthropic.com/v1 -> https://api.anthropic.com/v1/files)
    let base = api_endpoint.trim_end_matches('/').trim_end_matches("/messages");
    let files_url = format!("{}/files?beta=true", base);

    let file_part = multipart::Part::bytes(bytes)
        .file_name(filename.clone())
        .mime_str(content_type)?;
    let form = multipart::Form::new()
        .part("file", file_part);

    tracing::debug!("anthropic file upload: {url} -> {files_url}");
    let resp = http.post(&files_url)
        .header("x-api-key", api_key)
        .header("anthropic-version", ANTHROPIC_VERSION)
        .header("anthropic-beta", ANTHROPIC_FILES_BETA)
        .multipart(form)
        .send()
        .await?;

    if !resp.status().is_success() {
        let status = resp.status();
        let text = resp.text().await.unwrap_or_default();
        anyhow::bail!("anthropic file upload failed {status}: {text}");
    }

    let uploaded: UploadedFile = resp.json().await?;
    tracing::debug!("anthropic file uploaded: {} -> {}", filename, uploaded.id);

    {
        let mut cache = FILE_CACHE.lock().unwrap();
        cache.insert(cache_key, (uploaded.id.clone(), Instant::now()));
    }

    Ok(uploaded.id)
}



async fn download_raw(http: &Client, url: &str) -> Result<Vec<u8>> {
    if let Some(path) = url.strip_prefix("file://") {
        return Ok(tokio::fs::read(path).await?);
    }
    let resp = http.get(url)
        .timeout(std::time::Duration::from_secs(30))
        .send()
        .await?
        .error_for_status()?;
    Ok(resp.bytes().await?.to_vec())
}
