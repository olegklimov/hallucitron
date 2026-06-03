import asyncio
import json
import os
import sys
import time

from hallucitron import hallu_call, hallu_providers, hallu_structs

try:
    import pytest
except ImportError:
    pytest = None

# Live tests: they hit the real provider APIs and cost money. Keys are read from the
# repo's .test_api_keys file (NOT from environment variables) and injected into the
# default config; a test is skipped under pytest when its provider's key is absent.
# Run a subset directly:  python hallucitron/test_python_version.py pdf_anthropic

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_ARENA = os.path.join(_REPO_ROOT, "arena")
_REQ_DUMP_DIR = os.path.join(_ARENA, "req")


def _read_test_api_keys():
    """Parse .test_api_keys (shell `export NAME=VALUE` lines) into a {NAME: value}
    dict. This is the ONLY source of keys for the tests -- environment variables are
    deliberately not consulted."""
    path = os.path.join(_REPO_ROOT, ".test_api_keys")
    keys = {}
    try:
        with open(path) as f:
            lines = f.readlines()
    except FileNotFoundError:
        return keys
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):]
        name, sep, value = line.partition("=")
        if not sep:
            continue
        keys[name.strip()] = value.strip().strip('"').strip("'")
    return keys


def _load_test_config():
    """Default config with api keys filled from .test_api_keys (not the environment)."""
    keys = _read_test_api_keys()
    cfg = hallu_providers.load_default_config(use_env_keys=False)
    for prov in cfg.providers.values():
        env_name = prov.get("api_key_env")
        if env_name and keys.get(env_name):
            prov["api_key"] = keys[env_name]
    return cfg


_TEST_KEYS = _read_test_api_keys()
_CONFIG = _load_test_config()

ANTHROPIC_MODEL = "claude-sonnet-4-6"
OPENAI_MODEL = "gpt-5.4"
XAI_MODEL = "grok-4.20-0309-reasoning"

_KEY_ENV = {"anthropic": "ANTHROPIC_API_KEY", "openai": "OPENAI_API_KEY", "xai": "XAI_API_KEY"}

MEME_IMAGE_URL = "https://images.techadvisor.com/cmsdata/slideshow/3634008/funny_tech_memes_8.jpg"


def prefill_request(model):
    return hallu_providers.prefill_request_with_model(_CONFIG, model)


def tokyo_pdf_path():
    return os.path.join(_REPO_ROOT, "testfiles", "tokyo_weather.pdf")


def big_system():
    return "You produce structured output. " + "Context. " * 1500


WEATHER_SCHEMA = {
    "type": "object",
    "properties": {
        "city": {"type": "string"},
        "temp_c": {"type": "number"},
        "conditions": {"type": "array", "items": {"type": "string"}},
        "jacket": {"type": "boolean"},
    },
    "required": ["city", "temp_c", "conditions", "jacket"],
    "additionalProperties": False,
}


def pdf_messages():
    return [
        hallu_structs.HalluMessage("system", big_system(), ""),
        hallu_structs.HalluMessage("user", [
            {"m_type": "text", "m_content": "Read the attached PDF about Tokyo weather."},
            {"m_type": "pdf", "m_content": "file://" + tokyo_pdf_path()},
        ], ""),
        hallu_structs.HalluMessage("cd_instruction", "Fill the schema from the PDF, be brief.", ""),
    ]


def image_messages():
    return [
        hallu_structs.HalluMessage("system", big_system(), ""),
        hallu_structs.HalluMessage("user", [
            {"m_type": "text", "m_content": "What does this meme say? Describe the text and the joke."},
            {"m_type": "image/jpeg", "m_content": MEME_IMAGE_URL},
        ], ""),
    ]


def toolcall_messages():
    return [
        hallu_structs.HalluMessage("system", "You are concise. When asked about weather, you MUST call get_weather and never guess. After receiving the tool result, summarize the weather in one sentence mentioning the temperature.", ""),
        hallu_structs.HalluMessage("user", "What's the weather like in Barcelona right now? Do I need a jacket?", ""),
    ]


WEATHER_TOOLS = [{
    "type": "function",
    "name": "get_weather",
    "description": "Get current weather for a city",
    "parameters": {
        "type": "object",
        "properties": {
            "location": {"type": "string", "description": "City name"},
        },
        "required": ["location"],
    },
}]


def fake_weather_result(location):
    return json.dumps({
        "location": location,
        "temperature_c": 22,
        "condition": "Partly cloudy",
        "feels_like_c": 20,
        "jacket_recommended": True,
    })


def assert_weather(r):
    assert isinstance(r.parsed, dict), "parsed not dict: %r" % r.parsed
    assert isinstance(r.parsed["city"], str), "city not string"
    assert r.parsed["temp_c"] == 45, "temp_c not 45: %r" % r.parsed["temp_c"]
    assert isinstance(r.parsed["conditions"], list), "conditions not list"
    assert isinstance(r.parsed["jacket"], bool), "jacket not bool"
    assert r.usage.prompt_noncached > 0 or r.usage.cache_read_input_tokens > 0, "no prompt tokens"
    assert r.usage.output_tokens > 0, "output_tokens=0"


def assert_meme_freeform(r):
    text = r.parsed if isinstance(r.parsed, str) else r.raw_text
    assert "ipad" in text.lower(), "expected 'ipad' in freeform response: %r" % text
    assert r.usage.prompt_noncached > 0 or r.usage.cache_read_input_tokens > 0, "no prompt tokens"
    assert r.usage.output_tokens > 0, "output_tokens=0"


def assert_toolcall_first(r):
    assert r.tool_calls, "expected tool calls, got none. raw_text=%r" % r.raw_text
    tc = r.tool_calls[0]
    assert tc.name == "get_weather", "expected get_weather, got %r" % tc.name
    loc = tc.arguments.get("location", "")
    assert "barcelona" in loc.lower(), "expected Barcelona in location, got %r" % loc


def assert_toolcall_final(r):
    assert not r.tool_calls, "expected no more tool calls in final response"
    text = r.parsed if isinstance(r.parsed, str) else r.raw_text
    assert text.strip(), "final response text is empty"
    lower = text.lower()
    assert "22" in lower or "jacket" in lower or "barcelona" in lower, \
        "final response doesn't reference tool result: %r" % text


def append_tool_round(req, r, execute_tool):
    tc_json = [{"id": tc.call_id, "type": "function", "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)}} for tc in r.tool_calls]
    req.messages.append(hallu_structs.HalluMessage("assistant", r.raw_text, "", tool_calls=tc_json, provider_specific_stuff=r.provider_specific_stuff))
    for tc in r.tool_calls:
        output = execute_tool(tc)
        req.messages.append(hallu_structs.HalluMessage("tool", output, "", call_id=tc.call_id))


class HalluLog:
    def __init__(self, provider_name, experiment_name):
        self.buf = []
        self.provider_name = provider_name
        self.experiment_name = experiment_name

    def write(self, s):
        self.buf.append(s)

    def log_result(self, label, r, elapsed):
        cost_note = ""
        if r.provider_cost_usd is not None:
            cost_note = " provider_cost=$%.6f our_cost=$%.6f" % (r.provider_cost_usd, r.coins / 1e6)
        print("%s %s %s %.1fs %d coins%s" % (self.experiment_name, self.provider_name, label, elapsed, r.coins, cost_note))
        self.write('%s model=%r stop=%r' % (label, r.actual_model, r.stop_reason))
        self.write('  provider_usage: %s' % json.dumps(r.provider_usage_json, separators=(",", ":")))
        self.write('  usage: %s' % r.usage)
        for line in r.price_breakdown:
            self.write('  price: %s' % line)
        self.write('  coins: %d' % r.coins)
        if r.provider_cost_usd is not None:
            self.write('  provider_cost_usd: %.6f' % r.provider_cost_usd)
        self.write('  result: %s' % json.dumps(r.parsed, separators=(",", ":")))
        self.write('')

    def save(self, filename):
        os.makedirs(_ARENA, exist_ok=True)
        path = os.path.join(_ARENA, filename)
        with open(path, "w") as f:
            f.write("\n".join(self.buf) + "\n")
        print("  saved %s" % path)


async def _test_pdf(provider, model):
    log = HalluLog(provider, "pdf")
    req = prefill_request(model)
    req.messages = pdf_messages()
    req.output_schema = WEATHER_SCHEMA
    req.output_schema_name = "weather"
    req.max_tokens = 512
    req.temperature = 0.0
    req.req_dump_path = os.path.join(_REQ_DUMP_DIR, "pdf_%s_python.req" % provider)
    t = time.monotonic()
    r1 = await hallu_call(req)
    assert_weather(r1)
    log.log_result("call1", r1, time.monotonic() - t)
    t = time.monotonic()
    r2 = await hallu_call(req)
    log.write("")
    log.log_result("call2", r2, time.monotonic() - t)
    assert r2.usage.cache_read_input_tokens > 0, "no cache read on call2: %s" % r2.usage
    log.save("pdf_%s_python.log" % provider)


async def _test_image(provider, model):
    log = HalluLog(provider, "image")
    req = prefill_request(model)
    req.messages = image_messages()
    req.max_tokens = 512
    req.temperature = 0.0
    req.req_dump_path = os.path.join(_REQ_DUMP_DIR, "image_%s_python.req" % provider)
    t = time.monotonic()
    r1 = await hallu_call(req)
    assert_meme_freeform(r1)
    log.log_result("call1", r1, time.monotonic() - t)
    t = time.monotonic()
    r2 = await hallu_call(req)
    log.write("")
    log.log_result("call2", r2, time.monotonic() - t)
    assert r2.usage.cache_read_input_tokens > 0, "no cache read on call2: %s" % r2.usage
    log.save("image_%s_python.log" % provider)


async def _test_toolcall(provider, model):
    log = HalluLog(provider, "toolcall")
    req = prefill_request(model)
    req.messages = toolcall_messages()
    req.tools = WEATHER_TOOLS
    req.max_tokens = 1024
    req.temperature = 0.0
    req.req_dump_path = os.path.join(_REQ_DUMP_DIR, "toolcall_%s_python.req" % provider)
    t = time.monotonic()
    r1 = await hallu_call(req)
    assert_toolcall_first(r1)
    log.log_result("step1_tool_request", r1, time.monotonic() - t)
    for tc in r1.tool_calls:
        log.write("  tool_call: %s(%s) call_id=%s" % (tc.name, json.dumps(tc.arguments), tc.call_id))
    append_tool_round(req, r1, lambda tc: fake_weather_result(tc.arguments.get("location", "Barcelona")))
    t = time.monotonic()
    r2 = await hallu_call(req)
    assert_toolcall_final(r2)
    log.write("")
    log.log_result("step2_final_answer", r2, time.monotonic() - t)
    log.write("  final_text: %s" % (r2.parsed if isinstance(r2.parsed, str) else r2.raw_text))
    log.save("toolcall_%s_python.log" % provider)


async def _test_toolcall_streaming(provider, model):
    log = HalluLog(provider, "toolcall_stream")
    req = prefill_request(model)
    req.messages = toolcall_messages()
    req.tools = WEATHER_TOOLS
    req.max_tokens = 1024
    req.temperature = 0.0
    req.streaming = True
    req.req_dump_path = os.path.join(_REQ_DUMP_DIR, "toolcall_stream_%s_python.req" % provider)
    t = time.monotonic()
    r1 = await hallu_call(req)
    assert_toolcall_first(r1)
    log.log_result("step1_tool_request", r1, time.monotonic() - t)
    for tc in r1.tool_calls:
        log.write("  tool_call: %s(%s) call_id=%s" % (tc.name, json.dumps(tc.arguments), tc.call_id))
    for line in r1.sse_log:
        log.write(line)
    append_tool_round(req, r1, lambda tc: fake_weather_result(tc.arguments.get("location", "Barcelona")))
    t = time.monotonic()
    r2 = await hallu_call(req)
    assert_toolcall_final(r2)
    log.write("")
    log.log_result("step2_final_answer", r2, time.monotonic() - t)
    log.write("  final_text: %s" % (r2.parsed if isinstance(r2.parsed, str) else r2.raw_text))
    for line in r2.sse_log:
        log.write(line)
    log.save("toolcall_%s_python_stream.log" % provider)


# name -> (provider, model, coroutine factory)
ALL_TESTS = {
    "pdf_anthropic":           ("anthropic", ANTHROPIC_MODEL, _test_pdf),
    "image_anthropic":         ("anthropic", ANTHROPIC_MODEL, _test_image),
    "toolcall_anthropic":      ("anthropic", ANTHROPIC_MODEL, _test_toolcall),
    "toolcall_anthropic_stream": ("anthropic", ANTHROPIC_MODEL, _test_toolcall_streaming),
    "pdf_openai":              ("openai", OPENAI_MODEL, _test_pdf),
    "image_openai":            ("openai", OPENAI_MODEL, _test_image),
    "toolcall_openai":         ("openai", OPENAI_MODEL, _test_toolcall),
    "toolcall_openai_stream":  ("openai", OPENAI_MODEL, _test_toolcall_streaming),
    "pdf_xai":                 ("xai", XAI_MODEL, _test_pdf),
    "image_xai":               ("xai", XAI_MODEL, _test_image),
    "toolcall_xai":            ("xai", XAI_MODEL, _test_toolcall),
    "toolcall_xai_stream":     ("xai", XAI_MODEL, _test_toolcall_streaming),
}


def _make_pytest(name, provider, model, fn):
    def test():
        if not _TEST_KEYS.get(_KEY_ENV[provider]):
            pytest.skip("%s not in .test_api_keys" % _KEY_ENV[provider])
        asyncio.run(fn(provider, model))
    test.__name__ = "test_" + name
    return test


# Generate pytest-discoverable test_* functions
for _name, (_prov, _model, _fn) in ALL_TESTS.items():
    globals()["test_" + _name] = _make_pytest(_name, _prov, _model, _fn)


def _main():
    filters = sys.argv[1:]
    tests = ALL_TESTS
    if filters:
        tests = {k: v for k, v in tests.items() if any(f in k for f in filters)}
    if not tests:
        print("no tests matched %r, available: %s" % (filters, ", ".join(ALL_TESTS)))
        sys.exit(1)
    for name, (provider, model, fn) in tests.items():
        print("\n=== %s ===" % name)
        asyncio.run(fn(provider, model))
    print("\nall %d tests passed" % len(tests))


if __name__ == "__main__":
    _main()
