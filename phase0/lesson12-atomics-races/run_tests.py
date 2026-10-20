#!/usr/bin/env python3

import pathlib
import os
import subprocess
import sys


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: python3 run_tests.py ./minigpu_tests")
        return 2

    binary = pathlib.Path(sys.argv[1]).resolve()

    if not binary.is_file():
        print(f"test binary does not exist: {binary}")
        return 2

    timeout_seconds = float(
        os.environ.get("MINIGPU_TEST_TIMEOUT", "30")
    )

    try:
        completed = subprocess.run(
            [str(binary)],
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired as error:
        if error.stdout:
            print(error.stdout, end="")
        if error.stderr:
            print(error.stderr, end="", file=sys.stderr)

        print(
            f"FAIL: tests exceeded {timeout_seconds:g} seconds; "
            "a barrier or queue may be deadlocked",
            file=sys.stderr,
        )
        return 124

    print(completed.stdout, end="")
    print(completed.stderr, end="", file=sys.stderr)
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
