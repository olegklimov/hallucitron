import os
import json
import logging
import dataclasses

from hallucitron import anthropic_non_streaming
from hallucitron import anthropic_streaming
from hallucitron import openai_non_streaming
from hallucitron import openai_streaming


logger = logging.getLogger("hallu")

_seq = 0


async def hallu_call(req):
    global _seq
    _seq += 1
    fid = "%d-%d" % (os.getpid(), _seq)
    body_path = "/tmp/hallu-error-%s-body.json" % fid
    ours = not req.req_dump_path   # capture the body so a failure is replayable; honor a caller's own path
    if ours:
        req.req_dump_path = body_path
    try:
        result = await _dispatch(req)
    except json.JSONDecodeError:
        # Strict-schema JSON is normally well-formed, but grok stochastically (~1%) appends a stray
        # token to an otherwise-complete object. A fresh call returns clean, so retry once rather
        # than recover heuristically. A second failure is real -- dump forensics and raise.
        logger.warning("structured JSON parse failed; retrying once")
        try:
            result = await _dispatch(req)
        except Exception:
            _dump_forensics(req, fid)
            raise
    except Exception:
        _dump_forensics(req, fid)
        raise
    if ours:
        _quiet_remove(body_path)  # success: nothing to investigate
    result.usage.input_images = _count_input_images(req.messages)
    _apply_prices(result, req.provm_prices)
    return result


async def _dispatch(req):
    model = req.provm_name
    is_openai_compat = any(model.startswith(p) for p in ("gpt-", "o1-", "o3-", "o4-", "grok-"))
    if model.startswith("claude-") and req.streaming:
        return await anthropic_streaming.anthropic_streaming_call(req)
    elif model.startswith("claude-"):
        return await anthropic_non_streaming.anthropic_non_streaming_call(req)
    elif is_openai_compat and req.streaming:
        return await openai_streaming.openai_streaming_call(req)
    elif is_openai_compat:
        return await openai_non_streaming.openai_non_streaming_call(req)
    raise ValueError("hallu_call: unknown model prefix for %r" % model)


def _quiet_remove(path):
    try:
        os.remove(path)
    except OSError:
        pass


def _dump_forensics(req, fid):
    # The body was dumped to req.req_dump_path by the provider mid-call; pair it with the raw
    # messages and creds so the failure replays offline via replay_dumped_request.
    msgs_path = "/tmp/hallu-error-%s-messages.json" % fid
    creds_path = "/tmp/hallu-error-%s-creds.json" % fid
    try:
        with open(msgs_path, "w") as f:
            json.dump([dataclasses.asdict(m) for m in req.messages], f, indent=2, ensure_ascii=False, default=str)
        with open(creds_path, "w") as f:
            json.dump({
                "prov_name": req.prov_name,
                "prov_endpoint": req.prov_endpoint,
                "prov_api_key": req.prov_api_key,
                "provm_name": req.provm_name,
            }, f, indent=2)
    except Exception as e:
        logger.warning("hallu_call: could not write forensics: %s", e)
        return
    logger.error(
        "hallu_call failed; saved forensics:\n  body:  %s\n  msgs:  %s\n  creds: %s\n"
        "reproduce: python -m hallucitron.replay_dumped_request %s %s",
        req.req_dump_path, msgs_path, creds_path, req.req_dump_path, creds_path,
    )


def _count_input_images(messages):
    n = 0
    for msg in messages:
        if isinstance(msg.content, list):
            for part in msg.content:
                t = part.get("m_type", "")
                if t.startswith("image/"):
                    n += 1
    return n


def _apply_prices(result, prices):
    coins, breakdown = _convolute_usage_with_prices(result.usage, prices)
    result.coins = coins
    result.price_breakdown = breakdown


def _convolute_usage_with_prices(usage, prices):
    breakdown = []
    pairs = [
        ("pp1000t_prompt", usage.prompt_noncached),
        ("pp1000t_prompt_text", usage.prompt_noncached),
        ("pp1000t_prompt_without_cache", usage.prompt_noncached),
        ("pp1000t_prompt_cached", usage.cache_read_input_tokens),
        ("pp1000t_cache_read", usage.cache_read_input_tokens),
        ("pp1000t_cache_creation", usage.cache_creation_input_tokens),
        ("pp1000t_completion", usage.output_tokens),
    ]
    coins = 0
    for key, tokens in pairs:
        if tokens == 0:
            continue
        p = prices.get(key)
        if p is not None:
            c = (tokens * int(p)) // 1000
            breakdown.append("%d = %d * %d / 1000 %s" % (c, p, tokens, key))
            coins += c
    tool_pairs = [
        ("pp1call_web_search", usage.call_web_search),
        ("pp1call_x_search", usage.call_x_search),
        ("pp1call_code_interpreter", usage.call_code_interpreter),
        ("pp1call_document_search", usage.call_document_search),
        ("pp1call_file_search", usage.call_file_search),
        ("pp1image", usage.input_images),
    ]
    for key, calls in tool_pairs:
        if calls == 0:
            continue
        p = prices.get(key)
        if p is not None:
            c = calls * int(p)
            breakdown.append("%d = %d * %d %s" % (c, calls, p, key))
            coins += c
    return coins, breakdown
