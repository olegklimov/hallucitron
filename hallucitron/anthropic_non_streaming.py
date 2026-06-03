import json
import logging

import httpx

from hallucitron import anthropic_adapt
from hallucitron.anthropic_streaming import _apply_anthropic_thinking
from hallucitron.hallu_structs import HalluStructuredRequest, HalluStructuredResult, HalluToolCall, HalluUsage, HalluApiError, dump_req_body


logger = logging.getLogger("hallu")


ANTHROPIC_VERSION = "2023-06-01"


async def anthropic_non_streaming_call(req):
    http = httpx.AsyncClient()
    try:
        adapted = await anthropic_adapt.adapt_messages(req.messages, http, req.prov_api_key, req.prov_endpoint)
        endpoint = req.prov_endpoint.rstrip("/") + "/messages"
        body = {
            "model": req.provm_name,
            "max_tokens": req.max_tokens,
            "system": adapted.system,
            "messages": adapted.messages,
            "cache_control": {"type": "ephemeral", "ttl": "1h"},
        }
        thinking_on = _apply_anthropic_thinking(req, body)
        if req.output_schema:
            assert req.output_schema.get("additionalProperties") is False, \
                "output_schema must have additionalProperties: false (anthropic requirement)"
            body["output_config"] = {
                "format": {
                    "type": "json_schema",
                    "schema": req.output_schema,
                },
            }
        if req.tools:
            anthropic_tools = []
            for t in req.tools:
                if t.get("type") == "function":
                    anthropic_tools.append({
                        "name": t["name"],
                        "description": t.get("description", ""),
                        "input_schema": t.get("parameters", {}),
                    })
                else:
                    anthropic_tools.append(t)
            body["tools"] = anthropic_tools
            if req.output_schema:
                body["tool_choice"] = {"type": "none"}  # prevent tool calls (removing tools instead would break cache)
        if req.temperature is not None and not thinking_on:
            body["temperature"] = req.temperature
        headers = {
            "x-api-key": req.prov_api_key,
            "anthropic-version": ANTHROPIC_VERSION,
            "content-type": "application/json",
        }
        if adapted.needs_files_beta:
            headers["anthropic-beta"] = anthropic_adapt.ANTHROPIC_FILES_BETA
        dump_req_body(req, body)
        resp = await http.post(endpoint, headers=headers, json=body, timeout=120)
        if resp.status_code >= 400:
            raise HalluApiError(resp.status_code, "anthropic API error %d: %s" % (resp.status_code, resp.text))
        data = resp.json()
        return _parse_anthropic_response(data, req)
    finally:
        await http.aclose()


def _parse_anthropic_response(resp, req):
    actual_model = resp.get("model", req.provm_name)
    stop_reason = resp.get("stop_reason", "")
    response_id = resp.get("id", "")
    raw_text = ""
    thinking_text = ""
    thinking_blocks = []
    tool_calls = []
    for block in resp.get("content", []):
        btype = block.get("type")
        if btype == "text":
            raw_text += block.get("text", "")
        elif btype == "thinking":
            t = block.get("thinking")
            s = block.get("signature")
            if not t or not s:
                logger.warning("anthropic thinking block missing fields: thinking=%r signature=%r keys=%s", t is not None, s is not None, list(block.keys()))
            if thinking_text:
                thinking_text += "\n"
            thinking_text += t or ""
            thinking_blocks.append({"type": "thinking", "thinking": t or "", "signature": s or ""})
        elif btype == "tool_use":
            tool_calls.append(HalluToolCall(
                call_id=block.get("id", ""),
                name=block.get("name", ""),
                arguments=block.get("input", {}),
            ))
    if not raw_text.strip() and not tool_calls:
        raise RuntimeError("anthropic_call: no text output and no tool calls, max_tokens=%d, stop_reason=%r, response=%s" % (req.max_tokens, stop_reason, json.dumps(resp)))
    if tool_calls:
        parsed = None if not raw_text.strip() else raw_text
    elif req.output_schema is None:
        parsed = raw_text
    else:
        parsed = json.loads(raw_text)
    usage = resp.get("usage", {})
    input_tokens = usage.get("input_tokens", 0)
    output_tokens = usage.get("output_tokens", 0)
    cache_creation = usage.get("cache_creation_input_tokens", 0)
    cache_read = usage.get("cache_read_input_tokens", 0)
    return HalluStructuredResult(
        parsed=parsed,
        raw_text=raw_text,
        thinking_text=thinking_text,
        provider_specific_stuff=thinking_blocks if thinking_blocks else None,
        tool_calls=tool_calls,
        usage=HalluUsage(
            prompt_noncached=input_tokens,
            output_tokens=output_tokens,
            cache_creation_input_tokens=cache_creation,
            cache_read_input_tokens=cache_read,
        ),
        actual_model=actual_model,
        stop_reason=stop_reason,
        response_id=response_id,
        provider_usage_json=usage,
    )
