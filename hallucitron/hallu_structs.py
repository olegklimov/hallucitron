import json
import os
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Callable, Awaitable, Union

log = logging.getLogger("hallu")


class HalluApiError(RuntimeError):
    def __init__(self, status_code: int, message: str):
        self.status_code = status_code
        super().__init__(message)

    @property
    def is_client_error(self):
        return 400 <= self.status_code < 500


@dataclass
class HalluMessage:
    role: str                                           # system, user, assistant, tool, cd_instruction, context_file, hint, diff, cork, title, plain_text
    content: Union[str, List[Dict[str, Any]], None]     # str or [{m_type, m_content, ...}]
    author_label: str                                   # name of the user, wrapped by the adapter, e.g. "Humberto" or "727893532, via Telegram"
    tool_calls: Optional[List[Dict[str, Any]]] = None   # [{id, type, function: {name, arguments}}]
    call_id: str = ""
    provider_specific_stuff: Any = None  # provider-native blocks to prepend in assistant messages (e.g. anthropic thinking blocks with signatures)
    debug_key: str = ""                  # xxxyyy:001:042 used for log prefixes


@dataclass
class HalluStructuredRequest:
    prov_name: str = ""                 # "openai", "anthropic", "xai"
    prov_endpoint: str = ""
    prov_api_key: str = ""
    provm_name: str = ""
    provm_prices: Dict[str, Any] = field(default_factory=dict)
    messages: List[HalluMessage] = field(default_factory=list)
    output_schema: Optional[Dict[str, Any]] = None
    output_schema_name: str = ""
    tools: Optional[List[Dict[str, Any]]] = None    # [{type: "function", name, description, parameters}]
    max_tokens: int = 4096
    reasoning_effort: Optional[str] = None       # pick one of modelcap_reasoning_effort or leave unset
    temperature: Optional[float] = None
    streaming: bool = False
    on_text_delta: Optional[Callable[[str], Awaitable[None]]] = None
    req_dump_path: str = ""


@dataclass
class HalluUsage:
    prompt_noncached: int = 0
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0
    output_tokens: int = 0
    including_reasoning_tokens: int = 0
    input_images: int = 0
    call_web_search: int = 0
    call_x_search: int = 0
    call_code_interpreter: int = 0
    call_document_search: int = 0
    call_file_search: int = 0

    def __str__(self):
        parts = ["prompt_noncached=%d output=%d" % (self.prompt_noncached, self.output_tokens)]
        if self.including_reasoning_tokens > 0:
            parts.append("reasoning=%d" % self.including_reasoning_tokens)
        if self.cache_creation_input_tokens > 0:
            parts.append("cache_creation=%d" % self.cache_creation_input_tokens)
        if self.cache_read_input_tokens > 0:
            parts.append("cache_read=%d" % self.cache_read_input_tokens)
        if self.input_images > 0:
            parts.append("images=%d" % self.input_images)
        if self.call_web_search > 0:
            parts.append("web_search=%d" % self.call_web_search)
        if self.call_x_search > 0:
            parts.append("x_search=%d" % self.call_x_search)
        if self.call_code_interpreter > 0:
            parts.append("code_interpreter=%d" % self.call_code_interpreter)
        if self.call_document_search > 0:
            parts.append("document_search=%d" % self.call_document_search)
        if self.call_file_search > 0:
            parts.append("file_search=%d" % self.call_file_search)
        return " ".join(parts)


@dataclass
class HalluToolCall:
    call_id: str
    name: str
    arguments: Dict[str, Any]


# XXX add error message, return error within structure, for example ran out of completion tokens,
# cannot parse json, still have usage, therefore should be there in this structure.
# Then fix scenarios

@dataclass
class HalluStructuredResult:
    parsed: Any = None
    raw_text: str = ""
    thinking_text: str = ""          # anthropic thinking block text (empty if thinking not enabled)
    provider_specific_stuff: Any = None  # provider-native blocks from the response (e.g. anthropic thinking+signature), copy into assistant HalluMessage for multi-turn
    tool_calls: List[HalluToolCall] = field(default_factory=list)
    usage: HalluUsage = field(default_factory=HalluUsage)
    coins: int = 0
    price_breakdown: List[str] = field(default_factory=list)
    provider_cost_usd: Optional[float] = None
    actual_model: str = ""
    stop_reason: str = ""
    response_id: str = ""
    provider_usage_json: Dict[str, Any] = field(default_factory=dict)
    sse_log: List[Dict[str, Any]] = field(default_factory=list)


def parse_structured_text(raw_text):
    # A strict-schema response is one clean JSON value, but grok stochastically (~1%) appends a stray
    # token after the otherwise-complete object -- a plain json.loads then raises "Extra data". Two such
    # failures in a row defeated hallu_call's retry-once and crashed the service, so read the leading
    # value with raw_decode and drop trailing junk. A genuinely truncated object still raises and retries.
    obj, _ = json.JSONDecoder().raw_decode(raw_text.lstrip())
    return obj


def dump_req_body(req, body):
    if not req.req_dump_path:
        return
    d = os.path.dirname(req.req_dump_path)
    if d:
        os.makedirs(d, exist_ok=True)
    with open(req.req_dump_path, "w") as f:
        json.dump(body, f, indent=2, ensure_ascii=False)
