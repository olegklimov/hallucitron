import logging

from hallucitron import anthropic_non_streaming
from hallucitron import anthropic_streaming
from hallucitron import openai_non_streaming
from hallucitron import openai_streaming


logger = logging.getLogger("hallu")


async def hallu_call(req):
    model = req.provm_name
    is_openai_compat = any(model.startswith(p) for p in ("gpt-", "o1-", "o3-", "o4-", "grok-"))
    if model.startswith("claude-") and req.streaming:
        result = await anthropic_streaming.anthropic_streaming_call(req)
    elif model.startswith("claude-"):
        result = await anthropic_non_streaming.anthropic_non_streaming_call(req)
    elif is_openai_compat and req.streaming:
        result = await openai_streaming.openai_streaming_call(req)
    elif is_openai_compat:
        result = await openai_non_streaming.openai_non_streaming_call(req)
    else:
        raise ValueError("hallu_call: unknown model prefix for %r" % model)
    result.usage.input_images = _count_input_images(req.messages)
    _apply_prices(result, req.provm_prices)
    return result


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
