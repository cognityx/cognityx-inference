# OpenRouter

Provider name: `openrouter`. Adapter family: OpenAI-compatible chat
completions. Default model: `openai/gpt-4.1-mini`; provider-prefixed model IDs
are significant.

Create a key using the
[OpenRouter quickstart](https://openrouter.ai/docs/quickstart), then configure
`OPENROUTER_API_KEY`.

```json
{"OPENROUTER_API_KEY": ""}
```

```bash
uv run cognityx-inference providers models --provider openrouter --refresh
uv run cognityx-inference providers test --provider openrouter
```

The adapter normalizes streaming, usage, finish reasons, tools, structured
output, and common sampling parameters. Actual support can vary by the
downstream model/provider selected by OpenRouter; declare model-specific
policies where strict preflight behavior is required. A context budget is used
only when configured.

See the
[OpenRouter error guide](https://openrouter.ai/docs/api/reference/errors-and-debugging)
for upstream routing failures. Account balance, provider routing, price, and
limits can change and are not estimated by Cognityx. Rotate the key, update the
secret source, restart, verify, and revoke the old key. Because OpenRouter may
route to third parties, approve the route and data terms before sending
sensitive data. Never paste a key into chat, source code, or a shell command
argument. Operational links were reviewed 2026-07-29.
