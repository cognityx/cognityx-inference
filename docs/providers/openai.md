# OpenAI

Provider name: `openai`. Adapter family: OpenAI-compatible chat completions.
Default model: `gpt-4.1-mini`; verify it with model discovery before production
use.

1. Create an API key in the [OpenAI platform](https://platform.openai.com/api-keys).
2. Export `OPENAI_API_KEY`, or store it as `OPENAI_API_KEY` in the configured
   secrets JSON file.
3. Run:

```json
{"OPENAI_API_KEY": ""}
```

```bash
uv run cognityx-inference providers status
uv run cognityx-inference providers models --provider openai --refresh
uv run cognityx-inference providers test --provider openai
```

The adapter supports streaming, stop conditions, seed, log probabilities,
tools, structured output, and common sampling parameters when the selected
model supports them. Configure a model profile when a verified context limit is
needed for local budget enforcement. A discovered model without a profile is
usable but its context is reported as unknown.

Typical failures are `authentication_failed`, `permission_denied`,
`rate_limited`, and `model_unavailable`. Confirm the exact model ID and project
permissions before changing request parameters. Review current
[pricing](https://openai.com/api/pricing/) and
[usage limits](https://platform.openai.com/settings/organization/limits);
these are account-specific and are not inferred by Cognityx. Rotate a key in
the platform, update the environment or secrets file, restart the service, and
revoke the old key. Do not send sensitive data unless your OpenAI agreement and
organization policy permit it. Never paste a key into chat, source code, or a
shell command argument. Operational links were reviewed 2026-07-29.
