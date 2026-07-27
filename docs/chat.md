# Interactive Chat Client

`cognityx-inference chat` is a local, conversation-aware terminal client. It
uses the active model's certified context limit, keeps recent turns, and
summarizes older history before the next request can exceed the safe budget.

## Start

No arguments are required:

```bash
uv run cognityx-inference chat
```

If no model is resident, load one from inside chat:

```text
You [no model]> /load Qwen/Qwen3-8B vllm int4
```

The prompt and reply label always show the active model:

```text
You [Qwen/Qwen3-8B]> Explain KV-cache precision.
Qwen/Qwen3-8B> KV-cache precision controls ...
```

You may also select initial settings:

```bash
uv run cognityx-inference chat \
  --model Qwen/Qwen3-8B \
  --backend vllm \
  --profile int4 \
  --temperature 0.6 \
  --top-p 0.95 \
  --top-k 20 \
  --min-p 0.05 \
  --max-tokens 512 \
  --log-probabilities \
  --top-log-probabilities 5 \
  --timeout 300 \
  --first-token-timeout 60 \
  --no-token-progress-timeout 30
```

## Context protection and compression

The client obtains the active model's certified context limit and counts the
backend-formatted input tokens before sending each request. It reserves room
for the normal completion and for a separate summary request. Older turns are
compressed before the request exceeds that safe budget.

Compression is incremental: when a large old section cannot fit in one summary
request, smaller sections are summarized first. The client never silently drops
conversation history. If safe compression is impossible, it asks you to reduce
`max_tokens`, clear the history, or start a new chat.

## Commands

```text
/load <model> [backend] [profile]
/history
/clear
/save [chat-id]
/load_chat <chat-id>
/autosave on|off
/settings
/settings temperature 0.7
/settings top_k 40
/settings top_p 0.95
/settings min_p 0.05
/settings max_tokens 512
/settings stop END
/settings stop clear
/settings seed 42
/settings log_probabilities on
/settings top_log_probabilities 5
/settings reasoning on
/settings show power off
/settings show cpu off
/settings show ram off
/settings perf off
/quit
```

`/autosave on` writes a new immutable revision after every completed turn.
When autosave is off, `/quit` asks whether to save unsaved work.

## Saved sessions

Chat sessions are stored through `cognityx-storage`, scoped to the CLI owner
(`--owner-id`, defaulting to `COGNITYX_INFERENCE_USER` or `local`). The logical
key format is:

```text
users/<owner>/inference/chats/<chat-id>/<revision>.json
```

Each revision contains the system prompt, summary, retained messages, selected
parameters, display settings, model/runtime identity and certified-profile
reference. `/load_chat` restores the latest revision.

## Per-reply report and telemetry

Every reply shows input, thinking, answer, completion and total token counts
where the backend provides them, plus generation time and TTFT. The session
resource monitor stays active until chat exits and reports cumulative average
and peak GPU memory/power, host RAM/CPU and client-process RAM/CPU.

Use `--windows-bridge-path` with a compatible Windows performance bridge to add
Windows-host and shared-GPU measurements. Fields unavailable from the active
platform are shown as unavailable rather than estimated.
