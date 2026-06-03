use std::collections::HashMap;

use anyhow::{Context, Result};
use serde::Deserialize;
use serde_json::{json, Map, Value};

use super::hallu_structs::HalluStructuredRequest;


// A hallucitron config is a set of providers, each owning a list of models, plus a
// per-model price/capability table. See providers_default.yaml for the on-disk shape.
//
// A real application using this library supplies its own config -- in a multi-tenant
// system each tenant has a separate config with its own providers and api keys.
// load_default_config() loads the built-in providers_default.yaml, with an explicit
// choice of whether api keys come from the environment or straight from the config.

const DEFAULT_CONFIG: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../providers_default.yaml");

#[derive(Clone, Deserialize)]
pub struct ProviderCfg {
    pub name: String,
    pub kind: String,            // adapter family: "openai", "anthropic", "xai"
    pub endpoint: String,
    #[serde(default)]
    pub api_key: String,
    #[serde(default)]
    pub api_key_env: String,
    #[serde(default)]
    pub models: Vec<String>,
    #[serde(default)]
    pub more_prices: Value,
}

#[derive(Deserialize)]
struct ConfigFile {
    #[serde(default)]
    providers: HashMap<String, ProviderCfg>,
    #[serde(default)]
    models: HashMap<String, Value>,
}

pub struct HalluConfig {
    pub providers: HashMap<String, ProviderCfg>,
    pub models: HashMap<String, Value>,
    owner: HashMap<String, String>, // model-name -> provider-id
}

impl HalluConfig {
    fn new(providers: HashMap<String, ProviderCfg>, models: HashMap<String, Value>) -> Self {
        let mut owner = HashMap::new();
        for (pid, prov) in &providers {
            for m in &prov.models {
                owner.insert(m.clone(), pid.clone());
            }
        }
        HalluConfig { providers, models, owner }
    }

    pub fn provider_for_model(&self, model: &str) -> Result<&ProviderCfg> {
        let pid = self.owner.get(model)
            .ok_or_else(|| anyhow::anyhow!("model {model:?} not owned by any provider"))?;
        Ok(&self.providers[pid])
    }

    /// Model price table with the owning provider's more_prices merged in.
    pub fn prices_for_model(&self, model: &str) -> Result<Value> {
        let prov = self.provider_for_model(model)?;
        let mut map: Map<String, Value> = match self.models.get(model) {
            Some(Value::Object(o)) => o.clone(),
            _ => Map::new(),
        };
        if let Value::Object(extra) = &prov.more_prices {
            for (k, v) in extra {
                map.insert(k.clone(), v.clone());
            }
        }
        Ok(Value::Object(map))
    }
}


/// Build a HalluConfig from already-parsed config data.
///
/// use_env_keys = false: api keys are taken verbatim from the config (the multi-tenant
///   case -- the caller injected the keys it wants).
/// use_env_keys = true:  each provider's api_key is filled from the env var named in
///   `api_key_env`, overriding whatever the config held. Use this only for local
///   runs/tests where keys live in the environment, not the config.
fn build_config(mut file: ConfigFile, use_env_keys: bool) -> HalluConfig {
    if use_env_keys {
        for prov in file.providers.values_mut() {
            if !prov.api_key_env.is_empty() {
                if let Ok(v) = std::env::var(&prov.api_key_env) {
                    prov.api_key = v;
                }
            }
        }
    }
    HalluConfig::new(file.providers, file.models)
}

/// Load a config from a YAML file. See build_config for use_env_keys semantics.
pub fn load_config(path: &str, use_env_keys: bool) -> Result<HalluConfig> {
    let text = std::fs::read_to_string(path).with_context(|| format!("read {path}"))?;
    let file: ConfigFile = serde_yaml::from_str(&text).with_context(|| format!("parse {path}"))?;
    Ok(build_config(file, use_env_keys))
}

/// Load the built-in providers_default.yaml. By default api keys come from the config
/// as-is; pass use_env_keys = true to fill them from the environment instead.
pub fn load_default_config(use_env_keys: bool) -> Result<HalluConfig> {
    load_config(DEFAULT_CONFIG, use_env_keys)
}


/// Build a request for `model`, resolving its owning provider from `config`.
pub fn prefill_request_with_model(config: &HalluConfig, model: &str) -> Result<HalluStructuredRequest> {
    let prov = config.provider_for_model(model)?;
    if prov.api_key.is_empty() {
        anyhow::bail!(
            "no api_key for provider of model {model:?} \
             (set it in the config, or load with use_env_keys = true)"
        );
    }
    let prices = config.prices_for_model(model)?;
    Ok(HalluStructuredRequest {
        prov_name: prov.kind.clone(),
        prov_endpoint: prov.endpoint.clone(),
        prov_api_key: prov.api_key.clone(),
        provm_name: model.to_string(),
        provm_prices: if prices.is_null() { json!({}) } else { prices },
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


// Test setup: load the default config with api keys read from the repo's .test_api_keys
// file (NOT from environment variables). dotenvy parses the `export NAME=VALUE` lines;
// we read those values back and inject them into the config, leaving the process env
// untouched as the source of truth.
#[cfg(test)]
pub fn load_test_config() -> HalluConfig {
    use std::collections::HashMap as Map2;
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../.test_api_keys");
    let keys: Map2<String, String> = dotenvy::from_path_iter(path)
        .map(|iter| iter.filter_map(|r| r.ok()).collect())
        .unwrap_or_default();
    let mut cfg = load_default_config(false).expect("load default config");
    for prov in cfg.providers.values_mut() {
        if !prov.api_key_env.is_empty() {
            if let Some(v) = keys.get(&prov.api_key_env) {
                prov.api_key = v.clone();
            }
        }
    }
    cfg
}

#[cfg(test)]
pub fn test_key_present(api_key_env: &str) -> bool {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../.test_api_keys");
    dotenvy::from_path_iter(path)
        .map(|iter| iter.filter_map(|r| r.ok()).any(|(k, v)| k == api_key_env && !v.is_empty()))
        .unwrap_or(false)
}
