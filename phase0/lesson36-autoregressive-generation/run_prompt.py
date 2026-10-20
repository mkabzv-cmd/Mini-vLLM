#!/usr/bin/env python3

"""Tokenize text, run the Lesson 36 C++ generator, and decode its IDs.

The C++ runtime intentionally does not implement tokenization yet. This
script is the small bridge between human-readable text and that runtime:

    prompt text -> Hugging Face tokenizer -> C++/Metal generator
                -> generated token IDs -> Hugging Face tokenizer -> text

Everything is loaded from local files. No network access is used.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Sequence


UINT32_MAX = (1 << 32) - 1


def nonnegative_integer(text: str) -> int:
    """Parse an integer that is allowed to be zero."""

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
    """Parse and validate a directory path."""

    path = Path(text).expanduser()
    if not path.is_dir():
        raise argparse.ArgumentTypeError(
            f"directory does not exist: {path}"
        )
    return path.resolve()


def existing_executable(text: str) -> Path:
    """Parse and validate the path to the C++ generation binary."""

    path = Path(text).expanduser()
    if not path.is_file():
        raise argparse.ArgumentTypeError(
            f"generator binary does not exist: {path}"
        )
    if not os.access(path, os.X_OK):
        raise argparse.ArgumentTypeError(
            f"generator binary is not executable: {path}"
        )
    return path.resolve()


def load_tokenizer(model_directory: Path):
    """Load only the tokenizer files already present in model_directory."""

    try:
        from transformers import AutoTokenizer
        from transformers import logging as transformers_logging
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "run_prompt.py requires the local 'transformers' Python "
            "package used in Lesson 35"
        ) from error

    transformers_logging.set_verbosity_error()
    return AutoTokenizer.from_pretrained(
        model_directory,
        local_files_only=True,
    )


def tokenize(tokenizer, prompt: str, add_special_tokens: bool) -> list[int]:
    """Convert the prompt into IDs and validate the C++ uint32 boundary."""

    token_ids = tokenizer.encode(
        prompt,
        add_special_tokens=add_special_tokens,
    )

    if not token_ids:
        raise RuntimeError(
            "the tokenizer produced no token IDs; provide a nonempty prompt "
            "or keep special-token insertion enabled"
        )

    for token_id in token_ids:
        if token_id < 0 or token_id > UINT32_MAX:
            raise RuntimeError(
                f"token ID {token_id} does not fit in a C++ uint32_t"
            )

    return [int(token_id) for token_id in token_ids]


def read_output_ids(path: Path) -> list[int]:
    """Read the generator's whitespace-separated uint32 token IDs."""

    if not path.is_file():
        raise RuntimeError(
            "the C++ generator succeeded but did not create its output-ID "
            f"file: {path}"
        )

    text = path.read_text(encoding="utf-8")
    fields = text.split()
    if not fields:
        raise RuntimeError("the C++ generator wrote an empty output-ID file")

    token_ids: list[int] = []
    for field in fields:
        try:
            token_id = int(field, 10)
        except ValueError as error:
            raise RuntimeError(
                "the output-ID file must contain only whitespace-separated "
                f"decimal integers; found {field!r}"
            ) from error

        if token_id < 0 or token_id > UINT32_MAX:
            raise RuntimeError(
                f"generated token ID {token_id} is outside uint32_t range"
            )
        token_ids.append(token_id)

    return token_ids


def run_generator(
    generator_binary: Path,
    model_directory: Path,
    mode: str,
    max_new_tokens: int,
    prompt_ids: Sequence[int],
) -> tuple[list[int], subprocess.CompletedProcess[str]]:
    """Invoke the C++ generator using Lesson 36's positional CLI."""

    with tempfile.TemporaryDirectory(
        prefix="lesson36-generation-"
    ) as temporary_directory:
        output_path = Path(temporary_directory) / "generated_ids.txt"
        command = [
            str(generator_binary),
            str(model_directory),
            mode,
            str(max_new_tokens),
            str(output_path),
            *(str(token_id) for token_id in prompt_ids),
        ]

        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
        )

        if completed.returncode != 0:
            details = [
                f"C++ generator exited with status {completed.returncode}"
            ]
            if completed.stdout.strip():
                details.append("stdout:\n" + completed.stdout.rstrip())
            if completed.stderr.strip():
                details.append("stderr:\n" + completed.stderr.rstrip())
            raise RuntimeError("\n\n".join(details))

        generated_ids = read_output_ids(output_path)

    return generated_ids, completed


def validate_generated_ids(
    prompt_ids: Sequence[int],
    full_ids: Sequence[int],
    max_new_tokens: int,
) -> list[int]:
    """Validate that the file contains prompt IDs followed by new IDs."""

    if len(full_ids) < len(prompt_ids):
        raise RuntimeError(
            "the output-ID file is shorter than the original prompt"
        )

    actual_prefix = list(full_ids[: len(prompt_ids)])
    if actual_prefix != list(prompt_ids):
        raise RuntimeError(
            "the output-ID file does not start with the original prompt IDs\n"
            f"expected prefix: {list(prompt_ids)}\n"
            f"actual prefix:   {actual_prefix}"
        )

    new_ids = list(full_ids[len(prompt_ids) :])
    if len(new_ids) > max_new_tokens:
        raise RuntimeError(
            "the generator returned more new tokens than requested: "
            f"requested at most {max_new_tokens}, received {len(new_ids)}"
        )

    return new_ids


def display_text(label: str, text: str) -> None:
    """Print text with escapes visible, so spaces/newlines are unambiguous."""

    print(f"{label}: {json.dumps(text, ensure_ascii=False)}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Tokenize a prompt with local Hugging Face files, run the "
            "Lesson 36 C++/Metal generator, and decode its output IDs."
        )
    )
    parser.add_argument(
        "model_directory",
        type=existing_directory,
        help="local Hugging Face model/tokenizer directory",
    )
    parser.add_argument(
        "generator_binary",
        type=existing_executable,
        help="compiled lesson36_generate binary",
    )
    parser.add_argument("prompt", help="text prompt to continue")
    parser.add_argument(
        "--max-new-tokens",
        type=nonnegative_integer,
        default=8,
        help="maximum number of new tokens (default: 8)",
    )
    parser.add_argument(
        "--mode",
        choices=("greedy", "sample"),
        default="greedy",
        help="C++ token-selection mode (default: greedy)",
    )
    parser.add_argument(
        "--no-special-tokens",
        action="store_true",
        help="do not let the tokenizer add its normal BOS token",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()

    try:
        tokenizer = load_tokenizer(args.model_directory)
        prompt_ids = tokenize(
            tokenizer,
            args.prompt,
            add_special_tokens=not args.no_special_tokens,
        )
        full_ids, completed = run_generator(
            args.generator_binary,
            args.model_directory,
            args.mode,
            args.max_new_tokens,
            prompt_ids,
        )
        new_ids = validate_generated_ids(
            prompt_ids,
            full_ids,
            args.max_new_tokens,
        )
    except (OSError, RuntimeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    print(f"mode: {args.mode}")
    display_text("prompt", args.prompt)
    print(f"prompt token IDs: {prompt_ids}")
    print(f"generated token IDs: {new_ids}")
    print(f"all token IDs: {full_ids}")
    display_text(
        "generated text",
        tokenizer.decode(
            new_ids,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        ),
    )
    display_text(
        "full text",
        tokenizer.decode(
            full_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        ),
    )

    if completed.stdout.strip():
        print("\nC++ timing/details:")
        print(completed.stdout.rstrip())
    if completed.stderr.strip():
        print("\nC++ diagnostics:", file=sys.stderr)
        print(completed.stderr.rstrip(), file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
