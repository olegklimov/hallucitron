use std::fmt::Write as FmtWrite;

use serde_json::{json, Value};
use super::hallu_structs::{HalluMessage, HalluStructuredResult, HalluToolCall};


pub fn msg(role: &str, text: &str) -> HalluMessage {
    HalluMessage { role: role.into(), content: json!(text), tool_calls: json!(null), call_id: String::new(), provider_specific_stuff: json!(null) }
}

pub fn pdf_url_msg(role: &str, text: &str, url: &str) -> HalluMessage {
    HalluMessage {
        role: role.into(),
        content: json!([
            {"m_type": "text", "m_content": text},
            {"m_type": "pdf", "m_content": url},
        ]),
        tool_calls: json!(null),
        call_id: String::new(),
        provider_specific_stuff: json!(null),
    }
}

pub fn tokyo_pdf_url() -> String {
    let p = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../testfiles/tokyo_weather.pdf");
    format!("file://{}", p.display())
}

pub fn weather_schema() -> Value {
    json!({
        "type": "object",
        "properties": {
            "city": {"type": "string"},
            "temp_c": {"type": "number"},
            "conditions": {"type": "array", "items": {"type": "string"}},
            "jacket": {"type": "boolean"}
        },
        "required": ["city", "temp_c", "conditions", "jacket"],
        "additionalProperties": false
    })
}

// Ready-made message vectors

// Big system prompt triggers prompt caching (needs >2048 tokens for anthropic)
fn big_system() -> String {
    format!("You produce structured output. {}", "Context. ".repeat(1500))
}

pub fn pdf_messages() -> Vec<HalluMessage> {
    vec![
        msg("system", &big_system()),
        pdf_url_msg("user", "Read the attached PDF about Tokyo weather.", &tokyo_pdf_url()),
        msg("cd_instruction", "Fill the schema from the PDF, be brief."),
    ]
}

pub fn image_messages() -> Vec<HalluMessage> {
    vec![
        msg("system", &big_system()),
        image_url_msg("user", "What does this meme say? Describe the text and the joke.", MEME_IMAGE_URL),
    ]
}

pub fn assert_meme_freeform(r: &HalluStructuredResult) {
    let text = r.parsed.as_str().unwrap_or(&r.raw_text);
    assert!(text.to_lowercase().contains("ipad"), "expected 'ipad' in freeform response: {text:?}");
    assert!(r.usage.prompt_noncached > 0 || r.usage.cache_read_input_tokens > 0, "no prompt tokens");
    assert!(r.usage.output_tokens > 0, "output_tokens=0");
}

const MEME_IMAGE_URL: &str = "https://images.techadvisor.com/cmsdata/slideshow/3634008/funny_tech_memes_8.jpg";

fn image_url_msg(role: &str, text: &str, url: &str) -> HalluMessage {
    HalluMessage {
        role: role.into(),
        content: json!([
            {"m_type": "text", "m_content": text},
            {"m_type": "image/jpeg", "m_content": url},
        ]),
        tool_calls: json!(null),
        call_id: String::new(),
        provider_specific_stuff: json!(null),
    }
}


pub fn assert_weather(r: &HalluStructuredResult) {
    let obj = r.parsed.as_object().expect("not object");
    assert!(obj["city"].is_string(), "city not string: {:?}", obj["city"]);
    assert!(obj["temp_c"].as_f64().unwrap() == 45.0, "temp_c not 45: {:?}", obj["temp_c"]);
    assert!(obj["conditions"].is_array(), "conditions not array: {:?}", obj["conditions"]);
    assert!(obj["jacket"].is_boolean(), "jacket not boolean: {:?}", obj["jacket"]);
    assert!(r.usage.prompt_noncached > 0 || r.usage.cache_read_input_tokens > 0, "no prompt tokens at all");
    assert!(r.usage.output_tokens > 0, "output_tokens=0");
}


fn req_dump_path(experiment: &str, provider_name: &str) -> String {
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../arena/req");
    std::fs::create_dir_all(&dir).ok();
    dir.join(format!("{experiment}_{provider_name}_rust.req")).to_string_lossy().into_owned()
}

// 4 experiment runners, parametrized by (model, provider_name)

pub async fn run_pdf(model: &str, provider_name: &str) {
    println!();
    let cfg = super::hallu_providers::load_test_config();
    let mut log = HalluLog::new(provider_name, "pdf");

    let mut req = super::hallu_providers::prefill_request_with_model(&cfg, model).expect("prefill failed");
    req.messages = pdf_messages();
    req.output_schema = weather_schema();
    req.output_schema_name = "weather".into();
    req.max_tokens = 512;
    req.temperature = Some(0.0);
    req.req_dump_path = req_dump_path("pdf", provider_name);

    let t = std::time::Instant::now();
    let r1 = super::hallu_call(&req).await.expect("call 1 failed");
    assert_weather(&r1);
    log.log_result("call1", &r1, t.elapsed().as_secs_f64());

    let t = std::time::Instant::now();
    let r2 = super::hallu_call(&req).await.expect("call 2 failed");
    log.write("");
    log.log_result("call2", &r2, t.elapsed().as_secs_f64());
    assert!(
        r2.usage.cache_read_input_tokens > 0,
        "no cache read on call2: {:?}", r2.usage,
    );

    log.save(&format!("pdf_{provider_name}_rust.log"));
}

pub async fn run_image(model: &str, provider_name: &str) {
    println!();
    let cfg = super::hallu_providers::load_test_config();
    let mut log = HalluLog::new(provider_name, "image");

    let mut req = super::hallu_providers::prefill_request_with_model(&cfg, model).expect("prefill failed");
    req.messages = image_messages();
    req.max_tokens = 512;
    req.temperature = Some(0.0);
    req.req_dump_path = req_dump_path("image", provider_name);

    let t = std::time::Instant::now();
    let r1 = super::hallu_call(&req).await.expect("call 1 failed");
    assert_meme_freeform(&r1);
    log.log_result("call1", &r1, t.elapsed().as_secs_f64());

    let t = std::time::Instant::now();
    let r2 = super::hallu_call(&req).await.expect("call 2 failed");
    log.write("");
    log.log_result("call2", &r2, t.elapsed().as_secs_f64());
    assert!(
        r2.usage.cache_read_input_tokens > 0,
        "no cache read on call2: {:?}", r2.usage,
    );

    log.save(&format!("image_{provider_name}_rust.log"));
}


// Tool call test data

pub fn weather_tools() -> Value {
    json!([{
        "type": "function",
        "name": "get_weather",
        "description": "Get current weather for a city",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {"type": "string", "description": "City name"}
            },
            "required": ["location"]
        }
    }])
}

fn fake_weather_result(location: &str) -> String {
    json!({
        "location": location,
        "temperature_c": 22,
        "condition": "Partly cloudy",
        "feels_like_c": 20,
        "jacket_recommended": true
    }).to_string()
}

pub fn toolcall_messages() -> Vec<HalluMessage> {
    vec![
        msg("system", "You are concise. When asked about weather, you MUST call get_weather and never guess. After receiving the tool result, summarize the weather in one sentence mentioning the temperature."),
        msg("user", "What's the weather like in Barcelona right now? Do I need a jacket?"),
    ]
}

fn assert_toolcall_first(r: &HalluStructuredResult) {
    assert!(!r.tool_calls.is_empty(), "expected tool calls, got none. raw_text={:?}", r.raw_text);
    let tc = &r.tool_calls[0];
    assert_eq!(tc.name, "get_weather", "expected get_weather call, got {:?}", tc.name);
    let loc = tc.arguments["location"].as_str().unwrap_or("");
    assert!(loc.to_lowercase().contains("barcelona"), "expected Barcelona in location, got {loc:?}");
}

fn assert_toolcall_final(r: &HalluStructuredResult) {
    assert!(r.tool_calls.is_empty(), "expected no more tool calls in final response");
    let text = r.parsed.as_str().unwrap_or(&r.raw_text);
    assert!(!text.trim().is_empty(), "final response text is empty");
    // model should mention temperature from the fake result
    let lower = text.to_lowercase();
    assert!(lower.contains("22") || lower.contains("jacket") || lower.contains("barcelona"),
        "final response doesn't reference tool result: {text:?}");
}


// Append assistant tool_calls + tool result messages to the request, like adv_adapt.py does
fn append_tool_round(req: &mut super::hallu_structs::HalluStructuredRequest, r: &HalluStructuredResult, execute_tool: impl Fn(&HalluToolCall) -> String) {
    // Assistant message with tool_calls (openai format: [{id, type, function: {name, arguments}}])
    let tc_json: Vec<Value> = r.tool_calls.iter().map(|tc| json!({
        "id": tc.call_id,
        "type": "function",
        "function": {"name": tc.name, "arguments": tc.arguments.to_string()},
    })).collect();
    req.messages.push(HalluMessage {
        role: "assistant".into(),
        content: json!(r.raw_text),
        tool_calls: json!(tc_json),
        call_id: String::new(),
        provider_specific_stuff: r.provider_specific_stuff.clone(),
    });
    // Tool result messages
    for tc in &r.tool_calls {
        let output = execute_tool(tc);
        req.messages.push(HalluMessage {
            role: "tool".into(),
            content: json!(output),
            tool_calls: json!(null),
            call_id: tc.call_id.clone(),
            provider_specific_stuff: json!(null),
        });
    }
}


pub async fn run_toolcall(model: &str, provider_name: &str) {
    println!();
    let cfg = super::hallu_providers::load_test_config();
    let mut log = HalluLog::new(provider_name, "toolcall");

    let mut req = super::hallu_providers::prefill_request_with_model(&cfg, model).expect("prefill failed");
    req.messages = toolcall_messages();
    req.tools = weather_tools();
    req.max_tokens = 1024;
    req.temperature = Some(0.0);
    req.req_dump_path = req_dump_path("toolcall", provider_name);

    let t = std::time::Instant::now();
    let r1 = super::hallu_call(&req).await.expect("toolcall step 1 failed");
    assert_toolcall_first(&r1);
    log.log_result("step1_tool_request", &r1, t.elapsed().as_secs_f64());
    for tc in &r1.tool_calls {
        log.write(&format!("  tool_call: {}({}) call_id={}", tc.name, tc.arguments, tc.call_id));
    }

    append_tool_round(&mut req, &r1, |tc| {
        fake_weather_result(tc.arguments["location"].as_str().unwrap_or("Barcelona"))
    });

    let t = std::time::Instant::now();
    let r2 = super::hallu_call(&req).await.expect("toolcall step 2 failed");
    assert_toolcall_final(&r2);
    log.write("");
    log.log_result("step2_final_answer", &r2, t.elapsed().as_secs_f64());
    log.write(&format!("  final_text: {}", r2.parsed.as_str().unwrap_or(&r2.raw_text)));

    log.save(&format!("toolcall_{provider_name}_rust.log"));
}


pub async fn run_toolcall_streaming(model: &str, provider_name: &str) {
    println!();
    let cfg = super::hallu_providers::load_test_config();
    let mut log = HalluLog::new(provider_name, "toolcall_stream");

    let mut req = super::hallu_providers::prefill_request_with_model(&cfg, model).expect("prefill failed");
    req.messages = toolcall_messages();
    req.tools = weather_tools();
    req.max_tokens = 1024;
    req.temperature = Some(0.0);
    req.streaming = true;
    req.req_dump_path = req_dump_path("toolcall_stream", provider_name);

    let t = std::time::Instant::now();
    let r1 = super::hallu_call(&req).await.expect("toolcall streaming step 1 failed");
    assert_toolcall_first(&r1);
    log.log_result("step1_tool_request", &r1, t.elapsed().as_secs_f64());
    for tc in &r1.tool_calls {
        log.write(&format!("  tool_call: {}({}) call_id={}", tc.name, tc.arguments, tc.call_id));
    }
    for line in &r1.sse_log {
        log.write(line);
    }

    append_tool_round(&mut req, &r1, |tc| {
        fake_weather_result(tc.arguments["location"].as_str().unwrap_or("Barcelona"))
    });

    let t = std::time::Instant::now();
    let r2 = super::hallu_call(&req).await.expect("toolcall streaming step 2 failed");
    assert_toolcall_final(&r2);
    log.write("");
    log.log_result("step2_final_answer", &r2, t.elapsed().as_secs_f64());
    log.write(&format!("  final_text: {}", r2.parsed.as_str().unwrap_or(&r2.raw_text)));
    for line in &r2.sse_log {
        log.write(line);
    }

    log.save(&format!("toolcall_{provider_name}_rust_stream.log"));
}


pub struct HalluLog {
    pub buf: String,
    provider_name: String,
    experiment_name: String,
}

impl HalluLog {
    pub fn new(provider_name: &str, experiment_name: &str) -> Self {
        Self { buf: String::new(), provider_name: provider_name.into(), experiment_name: experiment_name.into() }
    }

    // write to log file only
    pub fn write(&mut self, s: &str) {
        writeln!(self.buf, "{s}").ok();
    }

    // compact one-liner to stdout, full detail to log file
    pub fn log_result(&mut self, label: &str, r: &HalluStructuredResult, elapsed: f64) {
        let mut cost_note = String::new();
        if let Some(usd) = r.provider_cost_usd {
            cost_note = format!(" provider_cost=${:.6} our_cost=${:.6}", usd, r.coins as f64 / 1e6);
        }
        println!("{} {} {label} {:.1}s {} coins{cost_note}", self.experiment_name, self.provider_name, elapsed, r.coins);
        self.write(&format!("{label} model={:?} stop={:?}", r.actual_model, r.stop_reason));
        self.write(&format!("  provider_usage: {}", r.provider_usage_json));
        self.write(&format!("  usage: {}", r.usage));
        for line in &r.price_breakdown {
            self.write(&format!("  price: {line}"));
        }
        self.write(&format!("  coins: {}", r.coins));
        if let Some(usd) = r.provider_cost_usd {
            self.write(&format!("  provider_cost_usd: {:.6}", usd));
        }
        self.write(&format!("  result: {}", serde_json::to_string(&r.parsed).unwrap()));
        self.write("");
    }

    pub fn save(&self, filename: &str) {
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../arena");
        std::fs::create_dir_all(&dir).ok();
        let path = dir.join(filename);
        std::fs::write(&path, &self.buf).expect("failed to write log");
        println!("  saved {}", path.display());
    }
}


// Tool call + thinking: same as run_toolcall_streaming but with thinking enabled,
// verifying that thinking blocks survive the multi-turn tool-use round-trip via provider_specific_stuff.

pub async fn run_toolcall_streaming_thinking(model: &str, provider_name: &str) {
    println!();
    let cfg = super::hallu_providers::load_test_config();
    let mut log = HalluLog::new(provider_name, "toolcall_thinking_stream");

    let mut req = super::hallu_providers::prefill_request_with_model(&cfg, model).expect("prefill failed");
    req.messages = toolcall_messages();
    req.tools = weather_tools();
    req.max_tokens = 16384;
    req.streaming = true;
    req.reasoning_effort = "high".into();
    req.req_dump_path = req_dump_path("toolcall_thinking_stream", provider_name);

    // Step 1: model should think then call get_weather
    let t = std::time::Instant::now();
    let r1 = super::hallu_call(&req).await.expect("toolcall+thinking step 1 failed");
    assert_toolcall_first(&r1);
    log.log_result("step1_tool_request", &r1, t.elapsed().as_secs_f64());
    for tc in &r1.tool_calls {
        log.write(&format!("  tool_call: {}({}) call_id={}", tc.name, tc.arguments, tc.call_id));
    }
    log.write(&format!("  thinking_text_len: {}", r1.thinking_text.len()));
    // thinking blocks must be present and carried in provider_specific_stuff
    assert!(!r1.thinking_text.is_empty(), "step1: thinking_text is empty, expected reasoning");
    assert!(!r1.provider_specific_stuff.is_null(), "step1: provider_specific_stuff is null, thinking blocks lost");
    let pss = r1.provider_specific_stuff.as_array().unwrap();
    assert!(pss.iter().all(|b| b["type"] == "thinking" && !b["signature"].as_str().unwrap_or("").is_empty()),
        "step1: thinking blocks must have type=thinking and non-empty signature");
    for line in &r1.sse_log {
        log.write(line);
    }

    // Append tool round — this copies provider_specific_stuff into the assistant HalluMessage
    append_tool_round(&mut req, &r1, |tc| {
        fake_weather_result(tc.arguments["location"].as_str().unwrap_or("Barcelona"))
    });

    // Step 2: model reads tool result, responds with final answer
    let t = std::time::Instant::now();
    let r2 = super::hallu_call(&req).await.expect("toolcall+thinking step 2 failed");
    assert_toolcall_final(&r2);
    log.write("");
    log.log_result("step2_final_answer", &r2, t.elapsed().as_secs_f64());
    log.write(&format!("  final_text: {}", r2.parsed.as_str().unwrap_or(&r2.raw_text)));
    log.write(&format!("  thinking_text_len: {}", r2.thinking_text.len()));
    for line in &r2.sse_log {
        log.write(line);
    }

    log.save(&format!("toolcall_{provider_name}_rust_thinking_stream.log"));
}
