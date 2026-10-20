# Local model fixtures

Downloaded model files are intentionally excluded from Git.

The lessons currently use this tiny random Hugging Face checkpoint:

```text
repository:
hf-internal-testing/tiny-random-LlamaForCausalLM

revision:
9fb191250dd56d0ba7ec9785a025ed29c03d5998

local directory:
models/hf-tiny-random-llama/
```

Download that exact revision with your preferred Hugging Face tooling and
place its files in the local directory above before running Lessons 35–39.

The checkpoint contains random test weights and does not produce meaningful
language. It is used to verify model loading, Metal execution, generation and
KV-cached decoding.
