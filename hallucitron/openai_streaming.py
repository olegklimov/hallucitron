import json

import httpx

from hallucitron import openai_adapt
from hallucitron.hallu_structs import HalluStructuredRequest, HalluStructuredResult, HalluToolCall, HalluUsage, HalluApiError, dump_req_body


SSE_JUNK_KEYS = {
    "item_id", "output_index", "logprobs", "content_index",
    "parallel_tool_calls", "previous_response_id", "reasoning",
    "tool_choice", "tools", "top_p", "top_logprobs",
    "presence_penalty", "frequency_penalty", "prompt_cache_key",
    "max_tool_calls", "safety_identifier", "store", "metadata",
    "background", "truncation", "user", "object",
    "annotations", "service_tier", "instructions",
    "incomplete_details", "text",
    "created_at", "completed_at", "id",
}


def _strip_sse_junk(v):
    if isinstance(v, dict):
        for k in list(SSE_JUNK_KEYS):
            v.pop(k, None)
        to_remove = [k for k, val in v.items() if val is None]
        for k in to_remove:
            del v[k]
        for val in v.values():
            _strip_sse_junk(val)
    elif isinstance(v, list):
        for val in v:
            _strip_sse_junk(val)


async def openai_streaming_call(req):
    http = httpx.AsyncClient()
    try:
        inp = await openai_adapt.adapt_messages(req.messages, http, req.prov_api_key, req.prov_endpoint, req.prov_name)
        assert req.prov_endpoint, "prov_endpoint must be set"
        endpoint = req.prov_endpoint.rstrip("/") + "/responses"
        body = {
            "model": req.provm_name,
            "input": inp,
            "stream": True,
        }
        if req.output_schema:
            body["text"] = {
                "format": {
                    "type": "json_schema",
                    "name": req.output_schema_name,
                    "schema": req.output_schema,
                    "strict": True,
                },
            }
        if req.tools:
            body["tools"] = req.tools
            if req.output_schema:
                body["tool_choice"] = "none"  # prevent tool calls (removing tools instead would break cache)
        if req.max_tokens > 0:
            body["max_output_tokens"] = req.max_tokens
        if req.reasoning_effort:
            effort = "high" if req.reasoning_effort == "max" else req.reasoning_effort
            body["reasoning"] = {"effort": effort}
        if req.temperature is not None and (not req.reasoning_effort or req.reasoning_effort == "none"):
            body["temperature"] = req.temperature
        dump_req_body(req, body)
        log = ["openai_streaming: model=%r" % req.provm_name]
        async with http.stream(
            "POST",
            endpoint,
            headers={
                "Authorization": "Bearer %s" % req.prov_api_key,
                "content-type": "application/json",
            },
            json=body,
            timeout=120,
        ) as response:
            if response.status_code >= 400:
                text = await response.aread()
                raise HalluApiError(response.status_code, "openai-compat streaming API error %d: %s" % (response.status_code, text.decode()))
            text_pieces = []
            resp_obj = {}
            event_count = 0
            pending_tool_calls = []   # [(call_id, name, [arg_pieces])]
            buf = ""
            async for chunk in response.aiter_text():
                buf += chunk
                while "\n\n" in buf:
                    pos = buf.index("\n\n")
                    block = buf[:pos]
                    buf = buf[pos + 2:]
                    for line in block.split("\n"):
                        if not line.startswith("data: "):
                            continue
                        data = line[len("data: "):]
                        event_count += 1
                        try:
                            ev = json.loads(data)
                        except json.JSONDecodeError:
                            continue
                        ev_type = ev.get("type", "")
                        if ev_type == "response.output_text.delta":
                            d = ev.get("delta")
                            if d:
                                text_pieces.append(d)
                                if req.on_text_delta:
                                    await req.on_text_delta(d)
                            _strip_sse_junk(ev)
                            log.append("SSE #%d: %s" % (event_count, json.dumps(ev, separators=(",", ":"))))
                        elif ev_type == "response.function_call_arguments.delta":
                            d = ev.get("delta")
                            if d and pending_tool_calls:
                                pending_tool_calls[-1][2].append(d)
                            _strip_sse_junk(ev)
                            log.append("SSE #%d: %s" % (event_count, json.dumps(ev, separators=(",", ":"))))
                        elif ev_type == "response.output_item.added":
                            item = ev.get("item", {})
                            if item.get("type") == "function_call":
                                call_id = item.get("call_id", "")
                                name = item.get("name", "")
                                pending_tool_calls.append((call_id, name, []))
                            _strip_sse_junk(ev)
                            log.append("SSE #%d: %s" % (event_count, json.dumps(ev, separators=(",", ":"))))
                        elif ev_type == "response.completed":
                            resp_obj = ev.get("response", {})
                            _strip_sse_junk(ev)
                            log.append("SSE #%d: %s" % (event_count, json.dumps(ev, separators=(",", ":"))))
                        else:
                            _strip_sse_junk(ev)
                            log.append("SSE #%d: %s" % (event_count, json.dumps(ev, separators=(",", ":"))))
        raw_text = "".join(text_pieces)
        tool_calls = []
        for call_id, name, arg_pieces in pending_tool_calls:
            args_str = "".join(arg_pieces)
            arguments = json.loads(args_str) if args_str.strip() else {}
            tool_calls.append(HalluToolCall(call_id=call_id, name=name, arguments=arguments))
        if not raw_text.strip() and not tool_calls:
            raise RuntimeError(
                "openai_streaming: no text output and no tool calls after %d events, max_tokens=%d, resp=%s\n%s" % (event_count, req.max_tokens, json.dumps(resp_obj), "\n".join(log))
            )
        if tool_calls:
            parsed = None
        elif req.output_schema is None:
            parsed = raw_text
        else:
            parsed = json.loads(raw_text)
        actual_model = resp_obj.get("model", req.provm_name)
        stop_reason = resp_obj.get("status", "")
        response_id = resp_obj.get("id", "")
        usage = resp_obj.get("usage", {})
        provider_usage_json = usage
        input_tokens = usage.get("input_tokens", 0)
        cached_tokens = (usage.get("input_tokens_details") or {}).get("cached_tokens", 0)
        reasoning_tokens = (usage.get("output_tokens_details") or {}).get("reasoning_tokens", 0)
        cost_ticks = usage.get("cost_in_usd_ticks")
        provider_cost_usd = cost_ticks / 10_000_000_000.0 if cost_ticks is not None else None
        def tool_detail(name):
            return (usage.get("server_side_tool_usage_details") or {}).get(name, 0)
        log.append("usage: %s" % json.dumps(usage, separators=(",", ":")))
        return HalluStructuredResult(
            parsed=parsed,
            raw_text=raw_text,
            tool_calls=tool_calls,
            usage=HalluUsage(
                prompt_noncached=input_tokens - cached_tokens,
                output_tokens=usage.get("output_tokens", 0),
                including_reasoning_tokens=reasoning_tokens,
                cache_creation_input_tokens=0,
                cache_read_input_tokens=cached_tokens,
                call_web_search=tool_detail("web_search_calls"),
                call_x_search=tool_detail("x_search_calls"),
                call_code_interpreter=tool_detail("code_interpreter_calls"),
                call_document_search=tool_detail("document_search_calls"),
                call_file_search=tool_detail("file_search_calls"),
            ),
            provider_cost_usd=provider_cost_usd,
            actual_model=actual_model,
            stop_reason=stop_reason,
            response_id=response_id,
            provider_usage_json=provider_usage_json,
            sse_log=log,
        )
    finally:
        await http.aclose()
