import json
import logging

import httpx

from hallucitron import anthropic_adapt
from hallucitron.hallu_structs import HalluStructuredRequest, HalluStructuredResult, HalluToolCall, HalluUsage, HalluApiError, dump_req_body, parse_structured_text


logger = logging.getLogger("hallu")

ANTHROPIC_VERSION = "2023-06-01"

SSE_JUNK_KEYS = {"content_block", "index", "logprobs"}


def _apply_anthropic_thinking(req, body):
    if not req.reasoning_effort:
        return False
    if req.provm_name in {"claude-opus-4-6", "claude-sonnet-4-6"}:
        body["thinking"] = {"type": "adaptive"}
    else:
        if req.max_tokens < 2048:
            logger.warning("apply_anthropic_thinking: max_tokens=%d too low for thinking (budget_tokens must be >= 1024), skipping", req.max_tokens)
            return False
        budget = max(req.max_tokens // 2, 1024)
        body["thinking"] = {"type": "enabled", "budget_tokens": budget}
    if req.reasoning_effort != "high":
        output_config = body.setdefault("output_config", {})
        output_config["effort"] = req.reasoning_effort
    return True


def _strip_sse_junk(v):
    if isinstance(v, dict):
        for k in SSE_JUNK_KEYS:
            v.pop(k, None)
        to_remove = [k for k, val in v.items() if val is None]
        for k in to_remove:
            del v[k]
        for val in v.values():
            _strip_sse_junk(val)
    elif isinstance(v, list):
        for val in v:
            _strip_sse_junk(val)


async def anthropic_streaming_call(req):
    http = httpx.AsyncClient()
    try:
        adapted = await anthropic_adapt.adapt_messages(req.messages, http, req.prov_api_key, req.prov_endpoint)
        assert req.prov_endpoint, "prov_endpoint must be set"
        endpoint = req.prov_endpoint.rstrip("/") + "/messages"
        body = {
            "model": req.provm_name,
            "max_tokens": req.max_tokens,
            "system": adapted.system,
            "messages": adapted.messages,
            "stream": True,
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
        log = ["anthropic_streaming: model=%r" % req.provm_name]
        headers = {
            "x-api-key": req.prov_api_key,
            "anthropic-version": ANTHROPIC_VERSION,
            "content-type": "application/json",
        }
        if adapted.needs_files_beta:
            headers["anthropic-beta"] = anthropic_adapt.ANTHROPIC_FILES_BETA
        dump_req_body(req, body)
        async with http.stream("POST", endpoint, headers=headers, json=body, timeout=120) as response:
            if response.status_code >= 400:
                text = await response.aread()
                raise HalluApiError(response.status_code, "anthropic streaming API error %d: %s" % (response.status_code, text.decode()))
            blocks = {}   # index -> ("text", text) or ("tool_use", id, name, partial_json)
            usage_input = 0
            usage_output = 0
            usage_cache_creation = 0
            usage_cache_read = 0
            stop_reason = None
            actual_model = None
            response_id = None
            provider_usage_json = {}
            event_count = 0
            buf = ""
            async for chunk in response.aiter_text():
                buf += chunk.replace("\r\n", "\n")
                while "\n\n" in buf:
                    pos = buf.index("\n\n")
                    frame = buf[:pos]
                    buf = buf[pos + 2:]
                    event_type = ""
                    data_lines = []
                    for line in frame.split("\n"):
                        if line.startswith("event:"):
                            event_type = line[len("event:"):].strip()
                        elif line.startswith("data:"):
                            data_lines.append(line[len("data:"):].lstrip())
                    if not data_lines:
                        continue
                    data = "\n".join(data_lines)
                    if data == "[DONE]":
                        break
                    try:
                        v = json.loads(data)
                    except json.JSONDecodeError:
                        continue
                    event_count += 1
                    if event_type == "message_start":
                        msg = v.get("message", {})
                        actual_model = msg.get("model")
                        response_id = msg.get("id")
                    # usage
                    u = None
                    if "message" in v and "usage" in v["message"]:
                        u = v["message"]["usage"]
                    elif "usage" in v:
                        u = v["usage"]
                    if u:
                        provider_usage_json = u
                        if u.get("input_tokens", 0) > 0:
                            usage_input = u["input_tokens"]
                        if u.get("output_tokens", 0) > 0:
                            usage_output = u["output_tokens"]
                        if u.get("cache_creation_input_tokens", 0) > 0:
                            usage_cache_creation = u["cache_creation_input_tokens"]
                        if u.get("cache_read_input_tokens", 0) > 0:
                            usage_cache_read = u["cache_read_input_tokens"]
                    if "delta" in v and "stop_reason" in v["delta"]:
                        sr = v["delta"]["stop_reason"]
                        if sr:
                            stop_reason = sr
                    if event_type == "content_block_start":
                        index = v.get("index", 0)
                        block = v.get("content_block", {})
                        btype = block.get("type")
                        if btype == "text":
                            blocks[index] = ["text", ""]
                        elif btype == "tool_use":
                            blocks[index] = ["tool_use", block.get("id", ""), block.get("name", ""), ""]
                        elif btype == "thinking":
                            blocks[index] = ["thinking", "", ""]  # thinking_text, signature
                        _strip_sse_junk(v)
                        log.append("SSE #%d: %s" % (event_count, json.dumps(v, separators=(",", ":"))))
                    elif event_type == "content_block_delta":
                        index = v.get("index", 0)
                        delta = v.get("delta", {})
                        dtype = delta.get("type")
                        if dtype == "text_delta":
                            chunk_text = delta.get("text", "")
                            if index in blocks and blocks[index][0] == "text":
                                blocks[index][1] += chunk_text
                            if chunk_text and req.on_text_delta:
                                await req.on_text_delta(chunk_text)
                        elif dtype == "input_json_delta":
                            chunk_text = delta.get("partial_json", "")
                            if index in blocks and blocks[index][0] == "tool_use":
                                blocks[index][3] += chunk_text
                        elif dtype == "thinking_delta":
                            chunk_text = delta.get("thinking", "")
                            if index in blocks and blocks[index][0] == "thinking":
                                blocks[index][1] += chunk_text
                        elif dtype == "signature_delta":
                            chunk_text = delta.get("signature", "")
                            if index in blocks and blocks[index][0] == "thinking":
                                blocks[index][2] += chunk_text
                        _strip_sse_junk(v)
                        log.append("SSE #%d: %s" % (event_count, json.dumps(v, separators=(",", ":"))))
                    elif event_type in ("content_block_stop", "message_start", "message_delta", "message_stop"):
                        _strip_sse_junk(v)
                        log.append("SSE #%d: %s" % (event_count, json.dumps(v, separators=(",", ":"))))
                    elif event_type == "ping":
                        pass
                    else:
                        _strip_sse_junk(v)
                        log.append("SSE #%d: %s" % (event_count, json.dumps(v, separators=(",", ":"))))
        raw_text = ""
        thinking_text = ""
        thinking_blocks = []
        tool_calls = []
        for index in sorted(blocks.keys()):
            b = blocks[index]
            if b[0] == "text":
                raw_text += b[1]
            elif b[0] == "thinking":
                if not b[1] or not b[2]:
                    logger.warning("anthropic streaming thinking block incomplete: thinking_len=%d signature_len=%d", len(b[1]), len(b[2]))
                if thinking_text:
                    thinking_text += "\n"
                thinking_text += b[1]
                thinking_blocks.append({"type": "thinking", "thinking": b[1], "signature": b[2]})
            elif b[0] == "tool_use":
                partial_json = b[3]
                if not partial_json.strip():
                    arguments = {}
                else:
                    arguments = json.loads(partial_json)
                tool_calls.append(HalluToolCall(call_id=b[1], name=b[2], arguments=arguments))
        if not raw_text.strip() and not tool_calls:
            raise RuntimeError(
                "anthropic_streaming: no text output and no tool calls after %d events, max_tokens=%d, stop_reason=%r\n%s" % (event_count, req.max_tokens, stop_reason, "\n".join(log))
            )
        if tool_calls:
            parsed = None if not raw_text.strip() else raw_text
        elif req.output_schema is None:
            parsed = raw_text
        else:
            parsed = parse_structured_text(raw_text)
        log.append("usage: input=%d output=%d cache_creation=%d cache_read=%d" % (
            usage_input, usage_output, usage_cache_creation, usage_cache_read,
        ))
        return HalluStructuredResult(
            parsed=parsed,
            raw_text=raw_text,
            thinking_text=thinking_text,
            provider_specific_stuff=thinking_blocks if thinking_blocks else None,
            tool_calls=tool_calls,
            usage=HalluUsage(
                prompt_noncached=usage_input,
                output_tokens=usage_output,
                cache_creation_input_tokens=usage_cache_creation,
                cache_read_input_tokens=usage_cache_read,
            ),
            actual_model=actual_model or req.provm_name,
            stop_reason=stop_reason or "",
            response_id=response_id or "",
            provider_usage_json=provider_usage_json,
            sse_log=log,
        )
    finally:
        await http.aclose()
