# Google Gemini

Provider name: `gemini`. Adapter family: native Gemini `generateContent`.
Default model: `models/gemini-flash-latest`. The optional `gemini_openai` adapter
is disabled by default.

Create a key following the
[Gemini API key guide](https://ai.google.dev/gemini-api/docs/api-key), then
export `GEMINI_API_KEY` or use the same key name in the secrets file.

```json
{"GEMINI_API_KEY": ""}
```

```bash
uv run cognityx-inference providers models --provider gemini --refresh
uv run cognityx-inference providers test --provider gemini
uv run cognityx-inference providers test \
  --provider gemini --structured-output
```

The native adapter supports streaming, top-k, stop conditions, seed, tools,
structured JSON output, and vision capability declaration. Unsupported
parameters fail before a request. Use exact `models/...` IDs returned by
discovery. Context budgets require a configured model profile; discovery alone
does not manufacture a limit.

Common diagnostics include an invalid or restricted key, unavailable model,
rate limiting, and malformed provider responses. Consult current
[Gemini pricing and quotas](https://ai.google.dev/gemini-api/docs/pricing);
free and paid data-use terms differ, so confirm the applicable tier before
sending sensitive data. Rotate the key in Google AI Studio, update configuration,
restart Cognityx Inference, and revoke the old key. Never paste a key into chat,
source code, or a shell command argument. Operational links were reviewed
2026-07-29.
