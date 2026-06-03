use std::fmt;

use serde::{Deserialize, Serialize};
use serde_json::Value;


// Input: flexus messages, easily convertible from db rows

#[derive(Debug, Clone)]
pub struct HalluMessage {
    pub role: String,       // system, user, assistant, tool, cd_instruction, context_file, hint, diff, cork, title, plain_text, kernel
    pub content: Value,     // string or [{m_type, m_content, ...}]
    pub tool_calls: Value,  // null or [{id, type, function: {name, arguments}}]
    pub call_id: String,    // for tool/diff role
    pub provider_specific_stuff: Value,  // null or provider-native blocks to prepend in assistant messages (e.g. anthropic thinking blocks with signatures)
}

#[derive(Debug, Clone)]
pub struct HalluStructuredRequest {
    pub prov_name: String,              // "openai", "anthropic", "xai"
    pub prov_endpoint: String,          // override endpoint (empty = default)
    pub prov_api_key: String,
    pub provm_name: String,
    pub provm_prices: Value,            // pp1000t_* price map (see test_models.yaml), used for coin calculation
    pub messages: Vec<HalluMessage>,
    pub output_schema: Value,           // json schema for structured output
    pub output_schema_name: String,     // name for the schema (anthropic wants it)
    pub tools: Value,                   // null or [{type: "function", name, description, parameters}]
    pub max_tokens: u32,
    pub reasoning_effort: String,       // pick one of modelcap_reasoning_effort or leave empty
    pub temperature: Option<f64>,
    pub streaming: bool,                // true = use SSE streaming, false = single response
    pub delta_tx: Option<tokio::sync::mpsc::Sender<String>>,  // text deltas sent here during streaming
    pub req_dump_path: String,          // if non-empty, dump the first API request body as pretty JSON here
}

// Output

#[derive(Clone, Default, Serialize, Deserialize)]
pub struct HalluUsage {
    pub prompt_noncached: u64,          // tokens billed at full prompt price (excludes cache)
    pub cache_creation_input_tokens: u64,
    pub cache_read_input_tokens: u64,

    pub output_tokens: u64,
    pub including_reasoning_tokens: u64,   // reasoning tokens (subset of output_tokens)

    pub input_images: u64,             // number of images in input messages

    // xAI server-side tool calls, billed per call via pp1call_* prices
    pub call_web_search: u64,
    pub call_x_search: u64,
    pub call_code_interpreter: u64,
    pub call_document_search: u64,
    pub call_file_search: u64,
}

impl fmt::Display for HalluUsage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "prompt_noncached={} output={}", self.prompt_noncached, self.output_tokens)?;
        if self.including_reasoning_tokens > 0 {
            write!(f, " reasoning={}", self.including_reasoning_tokens)?;
        }
        if self.cache_creation_input_tokens > 0 {
            write!(f, " cache_creation={}", self.cache_creation_input_tokens)?;
        }
        if self.cache_read_input_tokens > 0 {
            write!(f, " cache_read={}", self.cache_read_input_tokens)?;
        }
        if self.input_images > 0 { write!(f, " images={}", self.input_images)?; }
        if self.call_web_search > 0 { write!(f, " web_search={}", self.call_web_search)?; }
        if self.call_x_search > 0 { write!(f, " x_search={}", self.call_x_search)?; }
        if self.call_code_interpreter > 0 { write!(f, " code_interpreter={}", self.call_code_interpreter)?; }
        if self.call_document_search > 0 { write!(f, " document_search={}", self.call_document_search)?; }
        if self.call_file_search > 0 { write!(f, " file_search={}", self.call_file_search)?; }
        Ok(())
    }
}

impl fmt::Debug for HalluUsage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}

#[derive(Debug, Clone)]
pub struct HalluToolCall {
    pub call_id: String,
    pub name: String,
    pub arguments: Value,
}

#[derive(Debug, Clone)]
pub struct HalluStructuredResult {
    pub parsed: Value,              // the structured output, parsed from json
    pub raw_text: String,           // raw text before parsing (useful for debugging)
    pub thinking_text: String,      // anthropic thinking block text (empty if thinking not enabled)
    pub provider_specific_stuff: Value,  // provider-native blocks from the response (e.g. anthropic thinking+signature), copy into assistant HalluMessage for multi-turn
    pub tool_calls: Vec<HalluToolCall>,
    pub usage: HalluUsage,
    pub coins: i64,
    pub price_breakdown: Vec<String>,    // per-line breakdown from convolute_usage_with_prices
    pub provider_cost_usd: Option<f64>,  // cost reported by provider API, if available
    pub actual_model: String,
    pub stop_reason: String,
    pub response_id: String,        // provider response id, needed for openai tool call continuation
    pub provider_usage_json: Value, // raw usage object from provider API, for debugging
    pub sse_log: Vec<String>,       // SSE event log, populated when streaming=true
}

