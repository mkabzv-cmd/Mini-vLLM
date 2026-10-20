#!/usr/bin/env python3

"""Generate an official Hugging Face greedy reference for Lesson 36.

This deliberately uses the same slow algorithm as our first C++ generator:
on every step it reruns the model over the complete sequence, then selects
the largest final-token logit. It is a correctness oracle, not our runtime.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


def nonnegative_integer(text: str) -> int:
    try:
        value = int(text)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            f"expected an integer, got {text!r}"
        ) from error

    if value < 0:
        raise argparse.ArgumentTypeError("value must be nonnegative")
    return value


def existing_directory(text: str) -> Path:
    path = Path(text).expanduser()
    if not path.is_dir():
        raise argparse.ArgumentTypeError(
            f"directory does not exist: {path}"
        )
    return path.resolve()


def load_local_model(model_directory: Path):
    """Load model and tokenizer without consulting the network."""

    try:
        import torch
        from safetensors.torch import load_file
        from transformers import AutoTokenizer
        from transformers import LlamaConfig, LlamaForCausalLM
        from transformers import logging as transformers_logging
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "reference_generation.py requires torch, safetensors, and "
            "transformers from the Lesson 35 Python environment"
        ) from error

    transformers_logging.set_verbosity_error()
    tokenizer = AutoTokenizer.from_pretrained(
        model_directory,
        local_files_only=True,
    )
    config = LlamaConfig.from_pretrained(
        model_directory,
        local_files_only=True,
    )
    model = LlamaForCausalLM(config)
    state = load_file(model_directory / "model.safetensors")
    model.load_state_dict(state, strict=True)
    model.eval()
    return torch, tokenizer, model


def greedy_generate(
    torch,
    model,
    prompt_ids: list[int],
    max_new_tokens: int,
    stop_at_eos: bool,
) -> tuple[list[int], bool]:
    """Recompute the entire prefix and choose argmax at every step."""

    full_ids = list(prompt_ids)
    eos_token_id = model.config.eos_token_id
    stopped_on_eos = False

    with torch.no_grad():
        for _ in range(max_new_tokens):
            input_ids = torch.tensor(
                [full_ids],
                dtype=torch.long,
            )
            final_logits = model(
                input_ids=input_ids,
                use_cache=False,
            ).logits[0, -1]
            next_id = int(torch.argmax(final_logits).item())
            full_ids.append(next_id)

            if stop_at_eos and next_id == eos_token_id:
                stopped_on_eos = True
                break

    return full_ids, stopped_on_eos


def write_ids(path: Path, token_ids: list[int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        " ".join(str(token_id) for token_id in token_ids) + "\n",
        encoding="utf-8",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Create an official Hugging Face greedy-generation reference "
            "using only a local model directory."
        )
    )
    parser.add_argument(
        "model_directory",
        type=existing_directory,
        help="local Hugging Face model/tokenizer directory",
    )
    parser.add_argument("prompt", help="text prompt to continue")
    parser.add_argument(
        "--max-new-tokens",
        type=nonnegative_integer,
        default=8,
        help="number of greedy tokens to append (default: 8)",
    )
    parser.add_argument(
        "--output-ids",
        type=Path,
        help="optional whitespace-separated full-sequence ID file",
    )
    parser.add_argument(
        "--no-special-tokens",
        action="store_true",
        help="do not let the tokenizer add its normal BOS token",
    )
    parser.add_argument(
        "--no-stop-at-eos",
        action="store_true",
        help="continue for exactly max-new-tokens even after model EOS",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()

    try:
        torch, tokenizer, model = load_local_model(
            args.model_directory
        )
        prompt_ids = [
            int(token_id)
            for token_id in tokenizer.encode(
                args.prompt,
                add_special_tokens=not args.no_special_tokens,
            )
        ]
        if not prompt_ids:
            raise RuntimeError("the tokenizer produced no token IDs")

        full_ids, stopped_on_eos = greedy_generate(
            torch,
            model,
            prompt_ids,
            args.max_new_tokens,
            stop_at_eos=not args.no_stop_at_eos,
        )
        new_ids = full_ids[len(prompt_ids) :]

        if args.output_ids is not None:
            write_ids(args.output_ids.expanduser(), full_ids)
    except (OSError, RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    print(f"prompt: {json.dumps(args.prompt, ensure_ascii=False)}")
    print(f"prompt token IDs: {prompt_ids}")
    print(f"generated token IDs: {new_ids}")
    print(f"all token IDs: {full_ids}")
    print(f"stopped on EOS: {stopped_on_eos}")
    print(
        "generated text: "
        + json.dumps(
            tokenizer.decode(
                new_ids,
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            ),
            ensure_ascii=False,
        )
    )
    print(
        "full text: "
        + json.dumps(
            tokenizer.decode(
                full_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            ),
            ensure_ascii=False,
        )
    )

    if args.output_ids is not None:
        print(f"wrote IDs: {args.output_ids.expanduser()}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
