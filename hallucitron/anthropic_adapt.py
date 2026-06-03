import logging
import os
import re
import time
import json
import httpx

from hallucitron.hallu_structs import HalluMessage


logger = logging.getLogger("hallu")

# Strip user written "🪪 [...]"
_ACTOR_LABEL_PREFIX_RE = re.compile(r"^\s*🪪\s*\[[^\]]*\]\s*\n?", re.MULTILINE)

ANTHROPIC_VERSION = "2023-06-01"
ANTHROPIC_FILES_BETA = "files-api-2025-04-14"

_file_cache = {}   # "endpoint:url" -> (file_id, ts)
CACHE_TTL_SECS = 24 * 3600


class AnthropicAdapted:
    def __init__(self):
        self.system = []        # [{type: "text", text: "...", cache_control?}]
        self.messages = []      # [{role, content}]
        self.needs_files_beta = False


async def adapt_messages(msgs: list[HalluMessage], http: httpx.AsyncClient, api_key: str, api_endpoint: str) -> AnthropicAdapted:
    a = AnthropicAdapted()
    for m in msgs:
        role = m.role
        if role == "system":
            blocks, used_files = await _content_to_anthropic_blocks(m.content, http, api_key, api_endpoint, "", m.debug_key)
            a.needs_files_beta |= used_files
            a.system.extend(blocks)
        elif role == "assistant":
            # provider_specific_stuff carries anthropic-native blocks (thinking+signature) that must go first
            blocks = []
            if m.provider_specific_stuff and isinstance(m.provider_specific_stuff, list):
                blocks.extend(m.provider_specific_stuff)
            content_blocks, used_files = await _content_to_anthropic_blocks(m.content, http, api_key, api_endpoint, "", m.debug_key)
            a.needs_files_beta |= used_files
            blocks.extend(content_blocks)
            if m.tool_calls:
                for tc in m.tool_calls:
                    f = tc["function"]
                    inp = json.loads(f.get("arguments", "{}"))
                    blocks.append({
                        "type": "tool_use",
                        "id": tc.get("id", ""),
                        "name": f.get("name", ""),
                        "input": inp,
                    })
            a.messages.append({"role": "assistant", "content": blocks})
        elif role in ("user", "context_file", "hint", "plain_text"):
            blocks, used_files = await _content_to_anthropic_blocks(m.content, http, api_key, api_endpoint, m.author_label, m.debug_key)
            assert blocks, "what's that (1)\n%s" % str(m.content)
            a.needs_files_beta |= used_files
            a.messages.append({"role": "user", "content": blocks})
        elif role in ("cd_instruction",):
            blocks, used_files = await _content_to_anthropic_blocks(m.content, http, api_key, api_endpoint, "", m.debug_key)
            assert blocks, "what's that (2)\n%s" % str(m.content)
            assert blocks[0].get("type") == "text", "cd_instruction first block not text: %s" % blocks[0]
            assert isinstance(blocks[0].get("text"), str), "cd_instruction first block text not str: %s" % blocks[0]
            blocks[0]["text"] = "\U0001f4bf " + blocks[0]["text"]
            a.needs_files_beta |= used_files
            a.messages.append({"role": "user", "content": blocks})
        elif role in ("tool", "diff"):
            text = _content_to_text(m.content)
            a.messages.append({
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": m.call_id, "content": text}],
            })
        elif role in ("title", "cork"):
            pass
    # logger.info("anthropic adapt_messages system=\n%s\nmessages=\n%s",
    #     json.dumps(a.system, indent=2, ensure_ascii=False),
    #     json.dumps(a.messages, indent=2, ensure_ascii=False),
    # )
    return a


async def _content_to_anthropic_blocks(content, http, api_key, api_endpoint, author_label: str, debug_key: str):
    used_files = False
    if content is None:
        return [], False
    if isinstance(content, str):
        if not content:
            return [], False
        content = _ACTOR_LABEL_PREFIX_RE.sub("", content)
        return [{"type": "text", "text": f"🪪 [{author_label}]\n{content}" if author_label else content}], False
    if isinstance(content, list):
        blocks = []
        if author_label:
            blocks.append({"type": "text", "text": f"🪪 [{author_label}]"})
        for part in content:
            m_type = part.get("m_type", "")
            m_content = part.get("m_content", "")
            if m_type == "text":
                m_content = _ACTOR_LABEL_PREFIX_RE.sub("", m_content)
                if m_content.strip():
                    blocks.append({"type": "text", "text": m_content})
            elif m_type.startswith("image/"):
                if m_content.startswith("/"):
                    m_content = os.getenv("FLEXUS_WEB_URL", "http://localhost:8008").rstrip("/") + m_content
                if m_content.startswith(("http://", "https://", "file://")):
                    file_id = await _upload_or_cached(http, api_key, api_endpoint, m_content, content_type=m_type)
                    used_files = True
                    blocks.append({
                        "type": "image",
                        "source": {"type": "file", "file_id": file_id},
                    })
                else:
                    blocks.append({
                        "type": "image",
                        "source": {"type": "base64", "media_type": m_type, "data": m_content},
                    })
            elif m_type == "pdf":
                file_id = await _upload_or_cached(http, api_key, api_endpoint, m_content)
                used_files = True
                blocks.append({
                    "type": "document",
                    "source": {"type": "file", "file_id": file_id},
                })
            else:
                if m_content:
                    blocks.append({"type": "text", "text": m_content})
                else:
                    logger.warning("%s anthropic skip empty fallback text block for m_type=%r", debug_key, m_type)
        return blocks, used_files
    logger.error("%s something horrible: %s" % (debug_key, type(content)))
    return [{"type": "text", "text": "unknown stuff :|"}], False


def _content_to_text(content):
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(p.get("m_content", "") for p in content)
    return str(content)



async def _cache_get(cache_key):
    if cache_key in _file_cache:
        file_id, ts = _file_cache[cache_key]
        if time.time() - ts < CACHE_TTL_SECS:
            return file_id
        del _file_cache[cache_key]
    return None


async def _cache_set(cache_key, file_id):
    _file_cache[cache_key] = (file_id, time.time())


# XXX we already fixed infinite files accumulation for openai/xAI in openai_adapt.py, do the same here

async def _upload_or_cached(http, api_key, api_endpoint, url, content_type="application/pdf"):
    cache_key = "%s:%s" % (api_endpoint, url)
    hit = await _cache_get(cache_key)
    if hit:
        return hit
    data = await _download_raw(http, url)
    filename = url.rsplit("/", 1)[-1] or "document.pdf"
    base = api_endpoint.rstrip("/").removesuffix("/messages")
    files_url = "%s/files?beta=true" % base
    resp = await http.post(
        files_url,
        headers={
            "x-api-key": api_key,
            "anthropic-version": ANTHROPIC_VERSION,
            "anthropic-beta": ANTHROPIC_FILES_BETA,
        },
        files={"file": (filename, data, content_type)},
    )
    resp.raise_for_status()
    file_id = resp.json()["id"]
    await _cache_set(cache_key, file_id)
    return file_id



async def _download_raw(http, url):
    if url.startswith("file://"):
        path = url[len("file://"):]
        with open(path, "rb") as f:
            return f.read()
    resp = await http.get(url, timeout=30)
    resp.raise_for_status()
    return resp.content
