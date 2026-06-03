use std::collections::HashMap;
use std::sync::Mutex;
use std::time::Instant;

use anyhow::Result;
use base64::Engine as _;
use reqwest::{multipart, Client};
use serde::Deserialize;
use serde_json::{json, Value};

use super::hallu_structs::HalluMessage;


static FILE_CACHE: std::sync::LazyLock<Mutex<HashMap<String, (String, Instant)>>> =
    std::sync::LazyLock::new(|| Mutex::new(HashMap::new()));

const CACHE_TTL_SECS: u64 = 24 * 3600;


#[derive(Deserialize)]
struct UploadedFile {
    id: String,
}


pub async fn adapt_messages(msgs: &[HalluMessage], http: &Client, api_key: &str, api_endpoint: &str, prov_name: &str) -> Result<Vec<Value>> {
    let mut out: Vec<Value> = Vec::new();

    for m in msgs {
        match m.role.as_str() {
            "system" => {
                let parts = content_to_openai_parts(&m.content, http, api_key, api_endpoint, prov_name).await?;
                if !parts.is_empty() {
                    out.push(json!({"role": "system", "content": parts}));
                }
            }
            "assistant" => {
                let text = content_to_text(&m.content);
                if !text.is_empty() {
                    out.push(json!({"role": "assistant", "content": text}));
                }
                if let Some(tool_calls) = m.tool_calls.as_array() {
                    for tc in tool_calls {
                        let f = &tc["function"];
                        out.push(json!({
                            "type": "function_call",
                            "call_id": tc["id"].as_str().unwrap_or(""),
                            "name": f["name"].as_str().unwrap_or(""),
                            "arguments": f["arguments"].as_str().unwrap_or("{}"),
                        }));
                    }
                }
            }
            "user" | "context_file" | "hint" | "plain_text" => {
                let parts = content_to_openai_parts(&m.content, http, api_key, api_endpoint, prov_name).await?;
                assert!(!parts.is_empty(), "what's that (7)\n{:?}", m.content);
                out.push(json!({"role": "user", "content": parts}));
            }
            "cd_instruction" => {
                let parts = content_to_openai_parts(&m.content, http, api_key, api_endpoint, prov_name).await?;
                assert!(!parts.is_empty(), "what's that (8)\n{:?}", m.content);
                out.push(json!({"role": "system", "content": parts}));
            }
            "title" | "cork" => {}
            "tool" | "diff" => {
                let text = content_to_text(&m.content);
                out.push(json!({
                    "type": "function_call_output",
                    "call_id": m.call_id,
                    "output": text,
                }));
            }
            _ => {}
        }
    }

    Ok(out)
}


async fn content_to_openai_parts(content: &Value, http: &Client, api_key: &str, api_endpoint: &str, prov_name: &str) -> Result<Vec<Value>> {
    match content {
        Value::Null => Ok(vec![]),
        Value::String(s) => {
            if s.is_empty() {
                Ok(vec![])
            } else {
                Ok(vec![json!({"type": "input_text", "text": s})])
            }
        }
        Value::Array(parts) => {
            let mut out = Vec::new();
            for part in parts {
                let m_type = part["m_type"].as_str().unwrap_or("");
                let m_content = part["m_content"].as_str().unwrap_or("");
                match m_type {
                    "text" => {
                        if !m_content.trim().is_empty() {
                            out.push(json!({"type": "input_text", "text": m_content}));
                        }
                    }
                    "pdf" => {
                        let file_id = upload_or_cached(http, api_key, api_endpoint, m_content, "application/pdf").await?;
                        out.push(json!({"type": "input_file", "file_id": file_id}));
                    }
                    t if t.starts_with("image/") => {
                        if m_content.starts_with("http://") || m_content.starts_with("https://") || m_content.starts_with("file://") {
                            if prov_name.eq_ignore_ascii_case("xai") {
                                // xai doesn't support file_id for images
                                let bytes = download_url(http, m_content).await?;
                                let b64 = base64::engine::general_purpose::STANDARD.encode(&bytes);
                                out.push(json!({"type": "input_image", "image_url": format!("data:{t};base64,{b64}")}));
                            } else {
                                let file_id = upload_or_cached(http, api_key, api_endpoint, m_content, t).await?;
                                out.push(json!({"type": "input_image", "file_id": file_id}));
                            }
                        } else {
                            out.push(json!({"type": "input_image", "image_url": format!("data:{t};base64,{m_content}")}));
                        }
                    }
                    _ => {
                        out.push(json!({"type": "input_text", "text": m_content}));
                    }
                }
            }
            Ok(out)
        }
        other => {
            tracing::error!("something horrible: {:?}", other);
            Ok(vec![json!({"type": "input_text", "text": "unknown stuff :|"})])
        }
    }
}


fn content_to_text(content: &Value) -> String {
    match content {
        Value::Null => String::new(),
        Value::String(s) => s.clone(),
        Value::Array(parts) => {
            parts.iter()
                .filter_map(|p| p["m_content"].as_str())
                .collect::<Vec<_>>()
                .join("\n")
        }
        other => other.to_string(),
    }
}


async fn upload_or_cached(http: &Client, api_key: &str, api_endpoint: &str, url: &str, content_type: &str) -> Result<String> {
    let cache_key = format!("{}:{}", api_endpoint, url);

    // check cache
    {
        let cache = FILE_CACHE.lock().unwrap();
        if let Some((file_id, ts)) = cache.get(&cache_key) {
            if ts.elapsed().as_secs() < CACHE_TTL_SECS {
                tracing::debug!("openai file cache hit: {url} -> {file_id}");
                return Ok(file_id.clone());
            }
        }
    }

    let bytes = download_url(http, url).await?;
    let filename = url.rsplit('/').next().unwrap_or("document.pdf").to_string();

    // derive files endpoint from api_endpoint (e.g. https://api.openai.com/v1 -> https://api.openai.com/v1/files)
    let base = api_endpoint.trim_end_matches('/').trim_end_matches("/responses");
    let files_url = format!("{}/files", base);

    let file_part = multipart::Part::bytes(bytes)
        .file_name(filename.clone())
        .mime_str(content_type)?;
    let form = multipart::Form::new()
        .text("purpose", "user_data")
        .part("file", file_part);

    tracing::debug!("openai file upload: {url} -> {files_url}");
    let resp = http.post(&files_url)
        .bearer_auth(api_key)
        .multipart(form)
        .send()
        .await?
        .error_for_status()?;
    let uploaded: UploadedFile = resp.json().await?;
    tracing::debug!("openai file uploaded: {} -> {}", filename, uploaded.id);

    // cache it
    {
        let mut cache = FILE_CACHE.lock().unwrap();
        cache.insert(cache_key, (uploaded.id.clone(), Instant::now()));
    }

    Ok(uploaded.id)
}


async fn download_url(http: &Client, url: &str) -> Result<Vec<u8>> {
    if let Some(path) = url.strip_prefix("file://") {
        Ok(tokio::fs::read(path).await?)
    } else {
        let resp = http.get(url)
            .timeout(std::time::Duration::from_secs(30))
            .send()
            .await?
            .error_for_status()?;
        Ok(resp.bytes().await?.to_vec())
    }
}
