# Groq

Provider name: `groq`. Adapter family: OpenAI-compatible chat completions.
Default model: `llama-3.1-8b-instant`.

Create a key in the [Groq console](https://console.groq.com/keys), then export
`GROQ_API_KEY` or store `GROQ_API_KEY` in the configured secrets file.

```json
{"GROQ_API_KEY": ""}
```

```bash
uv run cognityx-inference providers models --provider groq --refresh
uv run cognityx-inference providers capabilities \
  --provider groq --model llama-3.1-8b-instant
uv run cognityx-inference providers test --provider groq
```

Groq supports the common streaming and sampling interface, but the built-in
policy rejects `logprobs`, `top_logprobs`, `logit_bias`, and `n` when supplied.
Parameters that Groq accepts but ignores are omitted from the effective request.
This validation occurs before network access.

HTTP 401 is classified as `authentication_failed`; HTTP 403 is
`permission_denied`. A prior observed 403 occurred during model discovery,
before model validation or inference, so it demonstrated credential/account
authorization rejection—not a bad model or sampling parameter. A previously
rejected key should not be retried unless it was rotated or explicitly
reconfirmed.

Use the [Groq API reference](https://console.groq.com/docs/api-reference) for
current models and the console for account limits. Cognityx keeps account
limits separate from context windows and does not estimate remaining spend.
Rotate and revoke keys in the console, then restart the service. Review your
contract and data policy before sending sensitive content. Never paste a key
into chat, source code, or a shell command argument. Operational links were
reviewed 2026-07-29.
