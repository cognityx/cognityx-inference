# Design patterns

Only patterns with visible structural support are listed.

## Strategy

**Participants:** `ContextProvider`, `FolderContextProvider`, `CONTEXT_PROVIDERS`,
and `get_context_provider()` in `contexts.py`.

The application asks a provider to load named chunks without depending on folder
details. The registry permits later vector-database or RAG providers to implement the
same protocol. This is the clearest GoF Strategy implementation in the project.

## Adapter

**Participants:** `VLLMLLM` in `vllm_engine.py`, `LocalLLM` in `llm.py`, and the
engine-neutral calls in `app.py`.

`VLLMLLM` translates vLLM construction, streaming outputs, finish reasons, and cache
metrics into the shape consumed by the existing application and report pipeline. It
is an Adapter; no common base class is required because Python's structural typing is
used at the call sites.

## Factory Method — pattern-like

**Participants:** `select_model_loader()` and `ModelLoaderSelection` in
`model_selection.py`; `get_context_provider()` in `contexts.py`.

These functions centralize object/class selection from configuration or registry
state. They are factory functions rather than subclass-overridden GoF Factory Methods,
so the implementation is deliberately marked pattern-like.

## Observer — pattern-like

**Participants:** the `on_text` callbacks accepted by `LocalLLM.generate()` and
`VLLMLLM.generate()`, plus `TimingTextIteratorStreamer`.

Generation publishes decoded chunks to a callback while remaining independent of
terminal rendering. This resembles Observer, but supports one callback rather than a
general subscriber collection, so it is pattern-like rather than a full GoF Observer.

## Patterns not claimed

The interactive command `if` chain is not a Command-pattern object hierarchy, and
ordinary dataclasses are not Builders or Mementos. The application module coordinates
subsystems, but it is not presented as a formal Facade because it also owns user
interaction and workflow policy.
