# Cloudflare Workers AI

Provider name: `cloudflare`. Adapter family: Cloudflare's OpenAI-compatible
endpoint. Default model: `@cf/meta/llama-3.1-8b-instruct`.

Create an API token and obtain the account ID by following the
[Workers AI REST setup](https://developers.cloudflare.com/workers-ai/get-started/rest-api/).
Configure both `CLOUDFLARE_API_TOKEN` and `CLOUDFLARE_ACCOUNT_ID`.

```json
{"CLOUDFLARE_API_TOKEN": "", "CLOUDFLARE_ACCOUNT_ID": ""}
```

```bash
uv run cognityx-inference providers status
uv run cognityx-inference providers test --provider cloudflare
```

Catalog discovery is disabled by default because the inference-compatible
endpoint is not treated as a model catalog. Configure exact model IDs from
Cloudflare documentation. The adapter normalizes the common OpenAI-compatible
request and response where the selected Workers AI model supports it. Declare a
profile to enforce a known context limit.

If either the token or account ID is missing, status is `not_configured` and no
request is attempted. Check token permissions and account identity for
authentication/permission failures. See the
[OpenAI compatibility guide](https://developers.cloudflare.com/workers-ai/configuration/open-ai-compatibility/)
for current behavior and the Cloudflare dashboard for limits and billing.
Rotate the token, update configuration, restart, test, and revoke the old token.
Review Cloudflare location and data terms before sensitive workloads. Never
paste a token into chat, source code, or a shell command argument. Operational
links were reviewed 2026-07-29.
