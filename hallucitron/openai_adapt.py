import asyncio
import base64
import json
import logging
import os
import re
import time

import httpx

from hallucitron.hallu_structs import HalluMessage


logger = logging.getLogger("hallu")

# Strip user written "🪪 [...]"
_ACTOR_LABEL_PREFIX_RE = re.compile(r"^\s*🪪\s*\[[^\]]*\]\s*\n?", re.MULTILINE)

_file_cache = {}   # "endpoint:url" -> (file_id, ts)
CACHE_TTL_SECS = 24 * 3600


def _report_crash(task):
    if task.cancelled():
        return
    if task.exception():
        logger.error("background task crashed", exc_info=task.exception())

_cleanup_tasks: dict[str, asyncio.Task] = {}   # "endpoint:api_key_tail" -> running task
_cleanup_ts: dict[str, float] = {}


async def adapt_messages(msgs: list[HalluMessage], http: httpx.AsyncClient, api_key: str, api_endpoint: str, prov_name: str) -> list[dict]:
    _maybe_launch_cleanup(api_key, api_endpoint)
    out = []
    for m in msgs:
        role = m.role
        if role == "system":
            parts = await _content_to_openai_parts(m.content, http, api_key, api_endpoint, prov_name, "", m.debug_key)
            if parts:
                out.append({"role": "system", "content": parts})
        elif role == "assistant":
            text = _content_to_text(m.content)
            if text:
                out.append({"role": "assistant", "content": text})
            if m.tool_calls:
                for tc in m.tool_calls:
                    f = tc["function"]
                    out.append({
                        "type": "function_call",
                        "call_id": tc.get("id", ""),
                        "name": f.get("name", ""),
                        "arguments": f.get("arguments", "{}"),
                    })
        elif role in ("user", "context_file", "hint", "plain_text"):
            parts = await _content_to_openai_parts(m.content, http, api_key, api_endpoint, prov_name, m.author_label, m.debug_key)
            assert parts, "what's that (3)?\n%s" % str(m.content)
            out.append({"role": "user", "content": parts})
        elif role in ("cd_instruction"):
            parts = await _content_to_openai_parts(m.content, http, api_key, api_endpoint, prov_name, "", m.debug_key)
            assert parts, "what's that (4)?\n%s" % str(m.content)
            out.append({"role": "system", "content": parts})
        elif role in ("title", "cork"):
            pass
        elif role in ("tool", "diff"):
            text = _content_to_text(m.content)
            out.append({
                "type": "function_call_output",
                "call_id": m.call_id,
                "output": text,
            })
    # logger.info("openai adapt_messages ->\n%s", json.dumps(out, indent=2, ensure_ascii=False))
    return out


async def _content_to_openai_parts(content, http, api_key, api_endpoint, prov_name, author_label: str, debug_key: str):
    if content is None:
        return []
    if isinstance(content, str):
        if not content:
            return []
        content = _ACTOR_LABEL_PREFIX_RE.sub("", content)
        return [{"type": "input_text", "text": f"🪪 [{author_label}]\n{content}" if author_label else content}]
    if isinstance(content, list):
        out = []
        if author_label:
            out.append({"type": "input_text", "text": f"🪪 [{author_label}]"})
        for part in content:
            m_type = part.get("m_type", "")
            m_content = part.get("m_content", "")
            if m_type == "text":
                m_content = _ACTOR_LABEL_PREFIX_RE.sub("", m_content)
                if m_content.strip():
                    out.append({"type": "input_text", "text": m_content})
            elif m_type == "pdf":
                file_id = await _upload_or_cached(http, api_key, api_endpoint, m_content)
                out.append({"type": "input_file", "file_id": file_id})
            elif m_type.startswith("image/"):
                if m_content.startswith("/"):
                    m_content = os.getenv("FLEXUS_WEB_URL", "http://localhost:8008").rstrip("/") + m_content
                if m_content.startswith(("http://", "https://", "file://")):
                    if prov_name.lower() == "xai":
                        # xai doesn't support file_id for images
                        data = await _download_raw(http, m_content)
                        b64 = base64.b64encode(data).decode()
                        out.append({"type": "input_image", "image_url": "data:%s;base64,%s" % (m_type, b64)})
                    else:
                        file_id = await _upload_or_cached(http, api_key, api_endpoint, m_content, content_type=m_type)
                        out.append({"type": "input_image", "file_id": file_id})
                else:
                    out.append({"type": "input_image", "image_url": "data:%s;base64,%s" % (m_type, m_content)})
            else:
                out.append({"type": "input_text", "text": m_content})
        return out
    logger.error("%s something horrible: %s" % (debug_key, type(content)))
    return [{"type": "input_text", "text": "unknown stuff :|"}]


def _content_to_text(content):
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(p.get("m_content", "") for p in content)
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


async def _upload_or_cached(http, api_key, api_endpoint, url, content_type="application/pdf"):
    cache_key = "%s:%s" % (api_endpoint, url)
    hit = await _cache_get(cache_key)
    if hit:
        return hit
    data = await _download_raw(http, url)
    filename = url.rsplit("/", 1)[-1] or "document.pdf"
    base = api_endpoint.rstrip("/").removesuffix("/responses")
    files_url = "%s/files" % base
    resp = await http.post(
        files_url,
        headers={"Authorization": "Bearer %s" % api_key},
        files={"file": (filename, data, content_type)},
        # data={"purpose": "user_data"},
        data={"purpose": "assistants"},
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


CLEANUP_MIN_INTERVAL_SECS = 3600

def _maybe_launch_cleanup(api_key, api_endpoint):
    # Fire-and-forget cleanup, one in-flight per (endpoint, api_key), and at most once per hour.
    # Key by last 8 of api_key so the log message doesn't leak the full key.
    # Owns its own httpx.AsyncClient -- the caller's client gets closed as soon as the completion
    # returns, so reusing it here would RuntimeError after the warm-up sleep.
    key = "%s:%s" % (api_endpoint, api_key[-8:])
    existing = _cleanup_tasks.get(key)
    if existing is not None and not existing.done():
        return
    now = time.time()
    last = _cleanup_ts.get(key, 0.0)
    if now - last < CLEANUP_MIN_INTERVAL_SECS:
        return
    _cleanup_ts[key] = now
    logger.info("cleanup_old_files %s: launching", key)
    t = asyncio.create_task(cleanup_old_files(api_key, api_endpoint))
    t.add_done_callback(_report_crash)
    _cleanup_tasks[key] = t


async def cleanup_old_files(api_key, api_endpoint):
    # Providers bill per-GiB-per-day for files sitting in their storage
    await asyncio.sleep(120)  # let the actual call that started this cleanup complete without noise in logs
    key = "%s:%s" % (api_endpoint, api_key[-8:])
    base = api_endpoint.rstrip("/").removesuffix("/responses")
    headers = {"Authorization": "Bearer %s" % api_key}
    cutoff = int(time.time()) - CACHE_TTL_SECS - 3600
    deleted = 0
    kept = 0
    async with httpx.AsyncClient() as http:
        params = {"limit": 500}   # this translates to 500 files deleted every CLEANUP_MIN_INTERVAL_SECS which should be fast enough (multiplied by the number of processes)
        resp = await http.get("%s/files" % base, headers=headers, params=params, timeout=30)
        resp.raise_for_status()
        payload = resp.json()
        items = payload["data"]
        for f in items:
            file_id = f["id"]
            created_at = f["created_at"]
            if created_at >= cutoff:
                kept += 1
                continue
            try:
                logger.info("%s DELETE %s", key, file_id)
                d = await http.delete("%s/files/%s" % (base, file_id), headers=headers, timeout=30)
                d.raise_for_status()
                deleted += 1
            except httpx.HTTPError as e:
                logger.info("cleanup_old_files %s: DELETE %s failed", key, file_id)
    logger.info("cleanup_old_files %s: deleted=%d kept=%d cutoff_age=%ds", key, deleted, kept, CACHE_TTL_SECS)
    return deleted, kept

