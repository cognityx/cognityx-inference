# Anthropic

Provider name: `anthropic`. Adapter family: native Anthropic Messages API.
Default model: `claude-sonnet-4-20250514`; verify exact availability for your
account.

Create a key in [Anthropic settings](https://platform.claude.com/settings/keys)
and configure `ANTHROPIC_API_KEY`.

```json
{"ANTHROPIC_API_KEY": ""}
```

```bash
uv run cognityx-inference providers models --provider anthropic --refresh
uv run cognityx-inference providers test --provider anthropic
```

The native adapter supports streaming, stop sequences, top-k, tools, and common
sampling parameters supported by Messages. It translates system messages into
Anthropic's system field and normalizes input/output token usage and termination
reason. Unsupported parameters, including log probabilities, fail locally.
Context budgeting requires a configured model profile.

Use the [Anthropic API overview](https://platform.claude.com/docs/en/api/overview)
for current headers and model behavior. Authentication/permission errors should
be resolved by checking workspace access and rotating the key; model errors
should be resolved using discovery. Current usage tiers, rate limits, and
billing are account-specific and are not estimated by Cognityx. Update the
secret source, restart, test the new key, and revoke the old one. Review
Anthropic data terms and your policy before sensitive use. Never paste a key
into chat, source code, or a shell command argument. Operational links were
reviewed 2026-07-29.
