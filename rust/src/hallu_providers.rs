use std::collections::HashMap;

use anyhow::{Context, Result};
use serde::Deserialize;
use serde_json::{json, Value};

use super::hallu_structs::HalluStructuredRequest;


// Credentials come from env vars (see .test_api_keys.example); test_models.yaml maps each
// model to a provider, endpoint, the env var holding its key, and optional prices.

#[derive(Deserialize)]
struct ModelCfg {
    provider: String,
    endpoint: String,
    api_key_env: String,
    #[serde(default)]
    prices: Value,
}

#[derive(Deserialize)]
struct ModelsFile {
    models: HashMap<String, ModelCfg>,
}


pub fn models_path() -> String {
    std::env::var("HALLU_MODELS").unwrap_or_else(|_| "test_models.yaml".to_string())
}


pub fn prefill_request_with_model(model: &str) -> Result<HalluStructuredRequest> {
    let path = models_path();
    let text = std::fs::read_to_string(&path).with_context(|| format!("read {path}"))?;
    let cfg: ModelsFile = serde_yaml::from_str(&text).with_context(|| format!("parse {path}"))?;
    let m = cfg.models.get(model)
        .ok_or_else(|| anyhow::anyhow!("model {model:?} not in {path}"))?;
    let api_key = std::env::var(&m.api_key_env)
        .map_err(|_| anyhow::anyhow!("env var {} not set (try: source .test_api_keys)", m.api_key_env))?;
    Ok(HalluStructuredRequest {
        prov_name: m.provider.clone(),
        prov_endpoint: m.endpoint.clone(),
        prov_api_key: api_key,
        provm_name: model.to_string(),
        provm_prices: if m.prices.is_null() { json!({}) } else { m.prices.clone() },
        messages: Vec::new(),
        output_schema: Value::Null,
        output_schema_name: String::new(),
        tools: Value::Null,
        max_tokens: 4096,
        temperature: None,
        streaming: false,
        delta_tx: None,
        req_dump_path: String::new(),
        reasoning_effort: String::new(),
    })
}


// Test setup: load .test_api_keys into the process env and point HALLU_MODELS at the
// repo's test_models.yaml, both relative to the crate root.
#[cfg(test)]
pub fn load_test_env() {
    dotenvy::from_path(concat!(env!("CARGO_MANIFEST_DIR"), "/../.test_api_keys")).ok();
    if std::env::var("HALLU_MODELS").is_err() {
        std::env::set_var("HALLU_MODELS", concat!(env!("CARGO_MANIFEST_DIR"), "/../test_models.yaml"));
    }
}
