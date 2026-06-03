# hallucitron

A small, provider-agnostic layer for **structured** LLM calls against OpenAI, Anthropic,
and xAI (Grok). One neutral request/response shape, one `hallu_call` entry point, and
per-provider adapters underneath. Python and Rust ports with matching behavior.

Features:

- **Structured output** — pass a JSON schema, get a parsed object back.
- **Tool calls** — request and round-trip function/tool calls across turns.
- **Streaming** — SSE streaming with a text-delta callback; thinking blocks (with
  signatures) are preserved across tool-use turns.
- **Files** — images and PDFs are uploaded to the provider's Files API and cached
  in-process by URL; OpenAI-compatible providers also get periodic file cleanup.
- **Usage & cost** — detailed token/tool-call accounting plus an optional integer
  "coins" cost derived from a per-model price table.

## Layout

```
hallucitron/        Python package  (import hallucitron); live tests in test_python_version.py
rust/               Rust crate      (src/lib.rs + Cargo.toml)
testfiles/          shared fixtures (tokyo_weather.pdf)
providers_default.yaml   built-in providers -> models -> prices config
.test_api_keys.example   template for local API keys (real file is git-ignored)
```

## Configuration

A **config** is a set of *providers*, each owning a list of *models*, plus a per-model
price/capability table. `providers_default.yaml` is the built-in default. Each provider
carries a `kind` (adapter family: `openai`, `anthropic`, or `xai`), an `endpoint`, an
`api_key`, and the env var name (`api_key_env`) to read when keys come from the
environment. A model is looked up by name — the loader finds the single provider whose
`models` list contains it.

API keys can come from two places, chosen explicitly:

- **From the config** (`use_env_keys=False`, the default) — keys are taken verbatim from
  the config. This is the multi-tenant case: each tenant supplies its own config with its
  own providers and keys; the library never reads the environment.
- **From the environment** (`use_env_keys=True`) — each provider's `api_key` is filled
  from the env var named in its `api_key_env`, for local runs where keys live in env.

```python
import hallucitron as h

cfg = h.load_default_config(use_env_keys=True)        # built-in providers, keys from env
cfg = h.load_config("my_tenant.yaml")                 # a tenant's own config, keys in-file
```

```sh
cp .test_api_keys.example .test_api_keys   # then edit in your keys
```

The live tests read keys **only** from `.test_api_keys` (never from environment
variables) and inject them into the default config.

## Python

```sh
pip install -e .          # installs hallucitron + httpx + pyyaml
```

```python
import asyncio
import hallucitron as h

async def main():
    cfg = h.load_default_config(use_env_keys=True)
    req = h.prefill_request_with_model(cfg, "claude-sonnet-4-6")
    req.messages = [
        h.HalluMessage("system", "You are concise.", ""),
        h.HalluMessage("user", "Name three primary colors.", ""),
    ]
    req.max_tokens = 200
    r = await h.hallu_call(req)
    print(r.raw_text)
    print(r.usage, "coins:", r.coins)

asyncio.run(main())
```

For structured output set `req.output_schema` (a JSON schema with
`additionalProperties: false`) and `req.output_schema_name`; `r.parsed` is the decoded
object. For streaming set `req.streaming = True` and assign an async
`req.on_text_delta` callback.

## Rust

```sh
cd rust && cargo build
```

```rust
let cfg = hallucitron::hallu_providers::load_default_config(true)?;  // keys from env
let mut req = hallucitron::hallu_providers::prefill_request_with_model(&cfg, "claude-sonnet-4-6")?;
req.messages = vec![ /* HalluMessage values */ ];
req.max_tokens = 200;
let r = hallucitron::hallu_call(&req).await?;
println!("{}", r.raw_text);
```

Streaming text deltas are delivered through `req.delta_tx` (a
`tokio::sync::mpsc::Sender<String>`).

## Tests

The live tests hit real provider APIs and **cost money**. They read keys from
`.test_api_keys` (not the environment); a provider whose key is absent is skipped
(pytest) or errors (cargo). No `source` step is needed.

```sh
# Python (pytest skips providers whose key is missing from .test_api_keys)
pytest hallucitron/test_python_version.py
# or run a subset directly:
python hallucitron/test_python_version.py pdf_anthropic toolcall_openai

# Rust
cd rust && cargo test -- --nocapture
```

Logs and dumped request bodies are written under `arena/` (git-ignored).

## Security

`.test_api_keys` holds live secrets and is git-ignored — never commit it. If a key is ever
exposed, rotate it.

## License

MIT — see [LICENSE](LICENSE).
