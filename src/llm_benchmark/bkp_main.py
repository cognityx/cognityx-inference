"""Legacy single-file prototype retained for historical reference.

The supported executable delegates to :mod:`llm_benchmark.app` through
:mod:`llm_benchmark.main`; this module is not part of the public API.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL = "Qwen/Qwen3-8B"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL)

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL,
    torch_dtype="auto",
    device_map="auto",
)

prompt = "Explain why the sky appears blue in three simple sentences."

messages = [
    {"role": "user", "content": prompt}
]

text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)

inputs = tokenizer(text, return_tensors="pt").to(model.device)

outputs = model.generate(
    **inputs,
    max_new_tokens=150,
)

response = tokenizer.decode(
    outputs[0][inputs.input_ids.shape[1]:],
    skip_special_tokens=True,
)

print("\nResponse:\n")
print(response)









def main():
    """Print the legacy prototype greeting."""
    print("Hello from llm-benchmark!")


if __name__ == "__main__":
    main()
