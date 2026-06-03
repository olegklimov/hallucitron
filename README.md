# Hallucitron

This project does the same as [litellm](https://github.com/BerriAI/litellm), but:

* Parallel code in Python and Rust
* Makes a best effort to reproduce provider pricing (Anthropic, xAI, OpenAI)
* Supports per-tenant configs for SaaS setups

Features:

* Anthropic API and Responses API (OpenAI, xAI and many others)
* Structured output
* Tool calls, strict and not strict
* Images
* PDFs


## Examples

Python:

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

The same thing in Rust:

```rust
use hallucitron::hallu_providers::{load_default_config, prefill_request_with_model};
use hallucitron::hallu_structs::HalluMessage;
use serde_json::json;

fn msg(role: &str, text: &str) -> HalluMessage {
    HalluMessage {
        role: role.into(),
        content: json!(text),
        tool_calls: json!(null),
        call_id: String::new(),
        provider_specific_stuff: json!(null),
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let cfg = load_default_config(true)?;  // keys from env
    let mut req = prefill_request_with_model(&cfg, "claude-sonnet-4-6")?;
    req.messages = vec![
        msg("system", "You are concise."),
        msg("user", "Name three primary colors."),
    ];
    req.max_tokens = 200;
    let r = hallucitron::hallu_call(&req).await?;
    println!("{}", r.raw_text);
    println!("{} coins: {}", r.usage, r.coins);
    Ok(())
}
```


## Testing

The tests hit real provider APIs and cost money, around $0.10 for a full run.

```sh
cp .test_api_keys.example .test_api_keys   # add your keys

pytest hallucitron/test_python_version.py                                 # all, skipping missing keys
python hallucitron/test_python_version.py pdf_anthropic toolcall_openai   # a subset

cd rust && cargo test -- --nocapture
```

Logs and dumped request bodies land in `arena/` (git-ignored).

`.test_api_keys` holds live secrets and is git-ignored. Never commit it. If one leaks, rotate it.


## License

MIT — see [LICENSE](LICENSE).
