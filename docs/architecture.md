# Architecture blueprint

## High-level module dependencies

```mermaid
flowchart LR
    main --> app
    app --> configuration
    app --> contexts
    app --> LocalLLM[llm.LocalLLM]
    app --> VLLMLLM[vllm_engine.VLLMLLM]
    app --> comparison
    app --> cost_estimation
    app --> download_preflight
    app --> loading_decision
    LocalLLM --> model_selection
    LocalLLM --> quantization
    LocalLLM --> model_loading
    LocalLLM --> model_diagnostics
    LocalLLM --> reporting
    VLLMLLM --> reporting
    VLLMLLM --> model_diagnostics
    model_diagnostics --> model_architecture
    model_diagnostics --> model_size
```

`main.py` modifies `sys.path` only when executed directly, then delegates immediately
to `app.main()`. Engine objects live for the duration of the interactive loop.

## Benchmark execution flow

```mermaid
sequenceDiagram
    actor User
    participant App as app.process_prompt
    participant Engine as LocalLLM or VLLMLLM
    participant Terminal
    participant Report as reporting
    User->>App: prompt or /benchmark
    App->>Engine: generate(prompt, on_text)
    loop decoded output
        Engine-->>Terminal: on_text(delta)
    end
    Engine-->>App: structured result
    App->>App: add context, cost, timestamp, system metadata
    App->>Terminal: metrics, JSON, Thinking, Answer
    App->>Report: save_benchmark_result(result)
    Report-->>App: collision-safe JSON path
    App-->>User: saved path and next prompt
```

## Model loading and quantization

```mermaid
flowchart TD
    Start[Model ID and requested profile] --> DryRun[Hub/local download preflight]
    DryRun --> Approval{Missing files approved?}
    Approval -- no --> Stop[Stop without loading]
    Approval -- yes --> Config[Load AutoConfig]
    Config --> Select[Select model architecture loader]
    Select --> Resolve{Resolve profile}
    Resolve --> Native[Use checkpoint-native quantization]
    Resolve --> BNB[Build BitsAndBytes INT4/INT8 config]
    Resolve --> Float[Choose BF16 or FP16 dtype]
    Native --> Load[from_pretrained]
    BNB --> Load
    Float --> Load
    Load --> Map[Inspect hf_device_map]
    Map --> Validate{Placement allowed?}
    Validate -- no --> Failure[Classify and save failure report]
    Validate -- yes --> Runtime[Collect load and runtime diagnostics]
```

Quantization resolution never silently converts an explicitly requested low-bit
profile to BF16. Checkpoint-native formats are distinguished from their effective
runtime path.

## Transformers and vLLM paths

```mermaid
flowchart TD
    CLI[--engine] --> Choice{engine}
    Choice -- transformers --> TEnv[Primary uv environment]
    TEnv --> TLoad[LocalLLM and Transformers generate]
    TLoad --> TStream[TextIteratorStreamer background thread]
    Choice -- vllm --> Detect{vLLM importable?}
    Detect -- no --> Reexec[Re-exec under .venv-vllm]
    Detect -- yes --> VLoad[VLLMLLM persistent offline engine]
    Reexec --> VLoad
    VLoad --> VStream[DELTA output from engine steps]
    TStream --> Schema[Shared result schema]
    VStream --> Schema
    Schema --> Output[Terminal report and JSON persistence]
```

The Transformers path supports interactive model switching. The vLLM path isolates
CUDA-sensitive dependencies and exposes normal/FP8 KV cache plus optional automatic
prefix caching.

## Context provider and corpus flow

```mermaid
flowchart LR
    Corpus[File, directory, or WikiText-103 Raw] --> Documents[corpus_documents]
    Documents --> Chunker[chunk_documents]
    Tokenizer --> Chunker
    Chunker --> JSONL[data/contexts/name/chunks.jsonl]
    JSONL --> Registry[CONTEXT_PROVIDERS]
    Registry --> Folder[FolderContextProvider]
    Folder --> Budget[build_context_prompt]
    Budget --> Complete[Append complete chunks]
    Complete --> Trim[Trim only final chunk if necessary]
    Trim --> Template[Engine chat template token count]
    Template --> Guard{prompt + output within limit?}
    Guard -- yes --> Generation
    Guard -- no --> ContextError
```

The prepared corpus remains separate from benchmark JSON. Results retain compact
context totals, a prompt preview, and a full-prompt hash.

## Important class relationships

```mermaid
classDiagram
    class GenerationSettings
    class LocalLLM {
      +load_model()
      +switch_model()
      +generate()
      +unload_model()
    }
    class VLLMLLM {
      +generate()
      +unload_model()
    }
    class ContextProvider {
      <<Protocol>>
      +load(name)
    }
    class FolderContextProvider
    class ModelLoadOptions
    class QuantizationResolution
    class ModelLoaderSelection
    class DownloadPreflight
    class CostConfig

    LocalLLM o-- GenerationSettings
    VLLMLLM o-- GenerationSettings
    LocalLLM --> ModelLoadOptions
    VLLMLLM --> ModelLoadOptions
    LocalLLM --> QuantizationResolution
    LocalLLM --> ModelLoaderSelection
    FolderContextProvider ..|> ContextProvider
    LocalLLM ..> DownloadPreflight
    VLLMLLM ..> DownloadPreflight
    CostConfig ..> LocalLLM : applied after result
```

`LocalLLM` and `VLLMLLM` intentionally use structural compatibility at the application
boundary rather than inheriting from a framework base class.
