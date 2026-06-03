import json

import httpx

from hallucitron import openai_adapt
from hallucitron.hallu_structs import HalluStructuredRequest, HalluStructuredResult, HalluToolCall, HalluUsage, HalluApiError, dump_req_body


async def openai_non_streaming_call(req):
    http = httpx.AsyncClient()
    try:
        inp = await openai_adapt.adapt_messages(req.messages, http, req.prov_api_key, req.prov_endpoint, req.prov_name)
        assert req.prov_endpoint, "prov_endpoint must be set"
        endpoint = req.prov_endpoint.rstrip("/") + "/responses"
        body = {
            "model": req.provm_name,
            "input": inp,
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
        resp = await http.post(
            endpoint,
            headers={
                "Authorization": "Bearer %s" % req.prov_api_key,
                "content-type": "application/json",
            },
            json=body,
            timeout=120,
        )
        if resp.status_code >= 400:
            raise HalluApiError(resp.status_code, "openai-compat API error %d: %s" % (resp.status_code, resp.text))
        data = resp.json()
        actual_model = data.get("model", req.provm_name)
        response_id = data.get("id", "")
        raw_text = ""
        tool_calls = []
        for item in data.get("output", []):
            itype = item.get("type")
            if itype == "message":
                for part in item.get("content", []):
                    if part.get("type") == "output_text":
                        raw_text += part.get("text", "")
            elif itype == "function_call":
                name = item.get("name", "")
                call_id = item.get("call_id", "")
                args_str = item.get("arguments", "{}")
                arguments = json.loads(args_str)
                tool_calls.append(HalluToolCall(call_id=call_id, name=name, arguments=arguments))
        if not raw_text.strip() and not tool_calls:
            raise RuntimeError("openai_responses_api: no text output and no tool calls, max_tokens=%d, response=%s" % (req.max_tokens, json.dumps(data)))
        if tool_calls:
            parsed = None
        elif req.output_schema is None:
            parsed = raw_text
        else:
            parsed = json.loads(raw_text)
        usage = data.get("usage", {})
        provider_usage_json = usage
        input_tokens = usage.get("input_tokens", 0)
        cached_tokens = (usage.get("input_tokens_details") or {}).get("cached_tokens", 0)
        reasoning_tokens = (usage.get("output_tokens_details") or {}).get("reasoning_tokens", 0)
        cost_ticks = usage.get("cost_in_usd_ticks")
        provider_cost_usd = cost_ticks / 10_000_000_000.0 if cost_ticks is not None else None
        def tool_detail(name):
            return (usage.get("server_side_tool_usage_details") or {}).get(name, 0)
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
            stop_reason=data.get("status", ""),
            response_id=response_id,
            provider_usage_json=provider_usage_json,
        )
    finally:
        await http.aclose()
