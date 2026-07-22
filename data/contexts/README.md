# Prepared contexts

Each folder contains a `chunks.jsonl` file and preparation manifest. Prepare
WikiText-103 Raw with the active model tokenizer using:

```bash
uv run python src/llm_benchmark/main.py \
  --model Qwen/Qwen3-32B \
  --prepare-context wikitext \
  --corpus wikitext-103-raw \
  --chunk-tokens 1024 \
  --chunk-overlap 128
```

The `datasets` library downloads `Salesforce/wikitext` using its normal cache;
the model-tokenized benchmark chunks are written to `data/contexts/wikitext/`.
