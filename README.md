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
hallucitron/        Python package  (import hallucitron)
rust/               Rust crate      (src/lib.rs + Cargo.toml)
tests/              Python live tests
testfiles/          shared fixtures (tokyo_weather.pdf)
test_models.yaml    model -> provider/endpoint/prices config
.test_api_keys.example   template for local API keys (real file is git-ignored)
```

## Configuration

Credentials come from environment variables; `test_models.yaml` maps each model to a
provider, endpoint, the env var holding its key, and optional prices.

```sh
cp .test_api_keys.example .test_api_keys   # then edit in your keys
source .test_api_keys
```

`HALLU_MODELS` overrides the config path (defaults to `test_models.yaml` in the cwd for
the library; the test suites point it at the repo copy automatically).

## Python

```sh
pip install -e .          # installs hallucitron + httpx + pyyaml
```

```python
import asyncio
import hallucitron as h

async def main():
    req = h.prefill_request_with_model("claude-sonnet-4-6")
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
let mut req = hallucitron::hallu_providers::prefill_request_with_model("claude-sonnet-4-6")?;
req.messages = vec![ /* HalluMessage values */ ];
req.max_tokens = 200;
let r = hallucitron::hallu_call(&req).await?;
println!("{}", r.raw_text);
```

Streaming text deltas are delivered through `req.delta_tx` (a
`tokio::sync::mpsc::Sender<String>`).

## Tests

The live tests hit real provider APIs and **cost money**. They need the matching API key
in the environment; without it they skip (pytest) or error (cargo).

```sh
source .test_api_keys

# Python (pytest skips providers whose key is unset)
pytest tests/
# or run a subset directly:
python tests/test_live.py pdf_anthropic toolcall_openai

# Rust
cd rust && cargo test -- --nocapture
```

Logs and dumped request bodies are written under `arena/` (git-ignored).

## Security

`.test_api_keys` holds live secrets and is git-ignored — never commit it. If a key is ever
exposed, rotate it.

## License

MIT — see [LICENSE](LICENSE).
