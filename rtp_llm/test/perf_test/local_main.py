"""Local Bazel test entry retaining benchmark outputs as test artifacts."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional, Sequence

from rtp_llm.test.perf_test.entry import main as perf_main


def main(argv: Optional[Sequence[str]] = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    original_directory = Path.cwd()
    # Bazel starts tests in runfiles. Resolve inputs there before moving to the
    # writable artifact directory; absolute output roots in JSON stay absolute.
    path_options = ("--config", "--run-dir", "--run-context")
    for index, argument in enumerate(arguments):
        if argument in path_options:
            if (
                index + 1 < len(arguments)
                and arguments[index + 1]
                and not arguments[index + 1].startswith("-")
            ):
                arguments[index + 1] = str(Path(arguments[index + 1]).absolute())
        else:
            option, separator, value = argument.partition("=")
            if option in path_options and separator and value:
                arguments[index] = option + "=" + str(Path(value).absolute())
    output_directory = os.environ.get("TEST_UNDECLARED_OUTPUTS_DIR")
    try:
        if output_directory:
            os.chdir(output_directory)
        return perf_main(arguments)
    finally:
        os.chdir(original_directory)


if __name__ == "__main__":
    raise SystemExit(main())
