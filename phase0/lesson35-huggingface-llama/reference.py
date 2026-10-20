#!/usr/bin/env python3

"""Generate Hugging Face reference logits for Lesson 35.

This script is only the correctness oracle. The miniature C++/Metal runtime
loads and executes the same checkpoint independently.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from safetensors.torch import load_file
from transformers import LlamaConfig, LlamaForCausalLM, logging


TOKEN_IDS = [1, 15043]  # Hugging Face tokenization of "Hello".

logging.set_verbosity_error()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_directory", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    config = LlamaConfig.from_pretrained(
        args.model_directory,
        local_files_only=True,
    )
    model = LlamaForCausalLM(config)
    state = load_file(
        args.model_directory / "model.safetensors"
    )
    model.load_state_dict(state, strict=True)
    model.eval()

    input_ids = torch.tensor(
        [TOKEN_IDS],
        dtype=torch.long,
    )

    with torch.no_grad():
        logits = model(
            input_ids=input_ids,
            use_cache=False,
        ).logits

    last_logits = (
        logits[0, -1]
        .detach()
        .cpu()
        .to(torch.float32)
        .numpy()
        .astype("<f4", copy=False)
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    last_logits.tofile(args.output)

    values, indices = torch.topk(logits[0, -1], 5)

    print(f"input IDs: {TOKEN_IDS}")
    print(f"logits shape: {tuple(logits.shape)}")
    print(f"top-5 token IDs: {indices.tolist()}")
    print(
        "top-5 logits: "
        + str([float(value) for value in values])
    )
    print(f"wrote: {args.output}")


if __name__ == "__main__":
    main()
