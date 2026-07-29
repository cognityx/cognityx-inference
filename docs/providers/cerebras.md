# Cerebras

Provider name: `cerebras`. Adapter family: OpenAI-compatible chat completions.
The example default and smoke-test model is `gpt-oss-120b`; confirm availability
for your account with discovery.

Create a key using the
[Cerebras API key console](https://inference-docs.cerebras.ai/console/api-keys)
and configure `CEREBRAS_API_KEY`.

```json
{"CEREBRAS_API_KEY": ""}
```

```bash
uv run cognityx-inference providers models --provider cerebras --refresh
uv run cognityx-inference providers test --provider cerebras
```

The common adapter exposes streaming, stop conditions, seed, log probabilities,
tools, structured output, and sampling controls subject to the selected model.
Use a model-specific profile for a verified context budget. Account quota and
model context are separate.

For 401/403 errors, rotate or review the key and account access. For
`model_unavailable`, select an exact ID returned by discovery. Follow the
[Cerebras quickstart](https://inference-docs.cerebras.ai/quickstart) and console
for current pricing and limits; Cognityx does not estimate balance. Rotate a
key by adding the replacement, restarting the service, testing it, and revoking
the old key. Review data terms before sensitive workloads. Never paste a key
into chat, source code, or a shell command argument. Operational links were
reviewed 2026-07-29.
