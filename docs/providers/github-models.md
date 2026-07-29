# GitHub Models

Provider name: `github_models`. Adapter family: GitHub Models with an
OpenAI-compatible inference transport and GitHub catalog discovery. Default
model: `openai/gpt-4.1-mini`.

Create a token with the permissions described in the
[GitHub Models quickstart](https://docs.github.com/en/github-models/quickstart)
and configure `GITHUB_MODELS_TOKEN`.

```json
{"GITHUB_MODELS_TOKEN": ""}
```

```bash
uv run cognityx-inference providers models \
  --provider github_models --refresh
uv run cognityx-inference providers test --provider github_models
```

The adapter adds GitHub media/API-version headers and uses exact catalog model
IDs. It normalizes common OpenAI-style streaming, usage, finish reasons,
sampling, tools, and structured output when the model supports them. Add a
profile for verified context budgeting.

Authentication and permission errors usually indicate token scope, organization
policy, or model access. Treat the GitHub Models free/preview experience as
evaluation unless your GitHub plan explicitly supports the intended production
use. Review the [GitHub Models documentation](https://docs.github.com/en/github-models/use-github-models)
for current limits and billing. Rotate the token, update configuration, restart,
test, then revoke the old token. Confirm repository, organization, and model
data policies before sensitive use. Never paste a token into chat, source code,
or a shell command argument. Operational links were reviewed 2026-07-29.
