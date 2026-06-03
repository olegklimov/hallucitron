"""
Replay a dumped provider request body. Used for reproducing failures offline.

Usage:
    python -m hallucitron.replay_dumped_request <body.json> <creds.json>

<body.json>  is the raw provider request body written by hallu_structs.dump_req_body
             (set HalluStructuredRequest.req_dump_path to capture one)
<creds.json> carries prov_endpoint + prov_api_key (+ prov_name, provm_name)

Picks /responses (openai-compat) or /messages (anthropic) based on prov_name in creds.
Streams raw SSE if streaming was enabled, otherwise prints the JSON response.
"""
import asyncio
import json
import sys

import httpx


async def _run(body_path, creds_path):
    with open(body_path) as f:
        body = json.load(f)
    with open(creds_path) as f:
        creds = json.load(f)
    endpoint = creds["prov_endpoint"].rstrip("/")
    api_key = creds["prov_api_key"]
    prov_name = creds.get("prov_name", "")
    if prov_name == "anthropic":
        url = endpoint + "/messages"
        headers = {
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
    else:
        url = endpoint + "/responses"
        headers = {
            "Authorization": "Bearer %s" % api_key,
            "content-type": "application/json",
        }
    streaming = bool(body.get("stream"))
    async with httpx.AsyncClient() as http:
        if streaming:
            async with http.stream("POST", url, headers=headers, json=body, timeout=180) as r:
                sys.stderr.write("HTTP %d\n" % r.status_code)
                async for chunk in r.aiter_text():
                    sys.stdout.write(chunk)
                    sys.stdout.flush()
        else:
            r = await http.post(url, headers=headers, json=body, timeout=180)
            sys.stderr.write("HTTP %d\n" % r.status_code)
            print(r.text)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.stderr.write("usage: python -m hallucitron.replay_dumped_request <body.json> <creds.json>\n")
        sys.exit(2)
    asyncio.run(_run(sys.argv[1], sys.argv[2]))
