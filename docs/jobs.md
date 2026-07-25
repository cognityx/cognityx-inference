# Discovery Jobs

## Why Jobs Exist

Hardware discovery can be long-running. It is better treated as a background
job than as a one-shot blocking request.

The current discovery path already supports:

- explicit job creation;
- live status lookup;
- server-sent event streaming;
- cancellation requests;
- persisted trial evidence;
- later reuse of the resulting certified profile.

## Ownership and Access

Every job has an immutable owner principal. Discovery list, status, events, and
cancel operations are scoped to that principal; a job owned by another caller
is reported as not found.

For a local single-user server, the owner defaults to `local` (or the value of
`COGNITYX_INFERENCE_LOCAL_PRINCIPAL`). For a shared server, configure bearer
keys outside the repository:

```bash
export COGNITYX_INFERENCE_API_KEYS_JSON='{"alice-key":"alice","bob-key":"bob"}'
export COGNITYX_INFERENCE_API_KEY='alice-key'  # client process
```

This is deliberately a minimal identity boundary. A future Cognityx auth and
authorization service can replace the resolver without changing job APIs.

This shape is intended to become the backbone for future Cognityx background
workloads such as ingestion, dataset preparation, and evaluation.

## Job Lifecycle

Discovery jobs move through these practical states:

- `queued`
- `running`
- `completed`
- `failed`
- `cancelled`
- `interrupted` (the inference server stopped before the in-process worker finished)

## Live Events

The service emits incremental events while a discovery runs:

- `discovery_started`
- `trial_started`
- `trial_completed`
- `cancel_requested`
- `discovery_completed`
- `discovery_failed`
- `discovery_cancelled`

This event stream is suitable for:

- CLI live viewing now;
- a future web UI later;
- external orchestration or monitoring layers.

Use `cognityx-inference discovery status` to list the current caller's active
jobs. Add `--all` for history, then use a returned ID with `watch` or `cancel`.

## Manual and Auto Discovery

The same discovery engine supports both explicit and automatic entry points.

- Manual discovery starts with `cognityx-inference discovery start`.
- Auto discovery starts from `model load` or `infer` when the caller chooses
  `--discovery-policy auto` or is otherwise asked to approve discovery.

Both paths create a normal owned job record, so the caller can still use the
same `status`, `watch`, and `cancel` commands even if the original shell is
gone.

When a job is running, the client can keep streaming progress while a second
terminal watches status. That is the intended day-to-day operator workflow.

## Current Implementation Notes

The current job-backed discovery flow is already useful operationally, but it
is still the first durable backbone step rather than the final general-purpose
job platform. In particular, discovery is the main integrated worker today, and
the broader shared background-job architecture will continue to expand from this
base.
