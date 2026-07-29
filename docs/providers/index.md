# Inference providers

Cognityx Inference separates the client-facing request contract from the service
provider and model. Applications use the same `CognityxInferenceClient` for a
local vLLM worker or a commercial API. A provider adapter translates only the
provider-specific transport and normalizes content, usage, finish reason,
latency, request identifiers, and available token details.

The built-in primary providers are `local`, `openai`, `groq`, `gemini`,
`cerebras`, `openrouter`, `github_models`, `cloudflare`, and `anthropic`.
`azure_openai`, `aws_bedrock`, and `google_vertex` are reserved extension
points and remain disabled until adapters are configured.

## Safe setup

Start by inspecting configuration requirements. This never prints credential
values:

```bash
uv run cognityx-inference providers setup
uv run cognityx-inference providers setup --create-template --yes
```

The template defaults to `.cognityx/secrets.json`, is created with mode `0600`
where supported, and contains empty placeholders. You can instead set
`secrets_file` in the inference TOML file or export
`COGNITYX_SECRETS_FILE=/absolute/path/to/secrets.json`. Environment variables
take precedence over the secrets file. Never commit the secrets file.

Check configuration without contacting unconfigured services:

```bash
uv run cognityx-inference providers list
uv run cognityx-inference providers status
uv run cognityx-inference providers status --json
```

Discover exact provider model identifiers and run a bounded smoke test:

```bash
uv run cognityx-inference providers models --provider openai --refresh
uv run cognityx-inference providers capabilities \
  --provider openai --model gpt-4.1-mini
uv run cognityx-inference providers test --provider openai
uv run cognityx-inference providers test --all
```

An absent credential is reported as `not_configured`; Cognityx does not attempt
the network. Provider errors are reduced to safe categories such as
`authentication_failed`, `permission_denied`, `rate_limited`,
`model_unavailable`, `network_error`, and `invalid_response`. Secret values,
authorization headers, and full upstream response bodies are never returned.

## Model profiles and limits

Provider definitions can declare model profiles as `unverified`, `discovered`,
`tested`, `certified`, or `deprecated`. Only a configured context limit is used to compute
a hard token budget. Discovery proves a model exists; it does not invent a
context window. Account quota and rate-limit information is reported separately
from model context and never interpreted as a token budget.

Parameter support is explicit per provider. Unsupported supplied parameters
fail locally before an API call. Provider-specific metadata remains in an
extension field without changing the common request contract.

## Routing and sensitive data

Purpose routes can name ordered provider/profile targets. Explicit caller
selection wins. Sensitive-data routes fail closed: if no allowed target is
available, the request is rejected instead of silently using a public provider.
External service suitability depends on your organization contract, region,
retention settings, and data policy; Cognityx does not make that decision for
you.

## Add an OpenAI-compatible provider

Add a disabled provider section with a unique name, `adapter =
"openai_compatible"`, its configured base URL, credential environment/secret
name, and an exact default model. Then define a `ParameterPolicy` preset only
after verifying which common fields the service accepts. Do not enable or ship
an untested preset. A future provider such as Together, Fireworks, DeepInfra,
SambaNova, or NVIDIA should require no new response model; a new native adapter
is warranted only when the wire protocol cannot be represented safely by the
existing compatible adapter.

See the individual guides:

- [OpenAI](openai.md)
- [Groq](groq.md)
- [Google Gemini](gemini.md)
- [Cerebras](cerebras.md)
- [OpenRouter](openrouter.md)
- [GitHub Models](github-models.md)
- [Cloudflare Workers AI](cloudflare.md)
- [Anthropic](anthropic.md)
