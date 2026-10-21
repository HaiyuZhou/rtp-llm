"""Sequential Kineto analysis in an exec-isolated, externally bounded process."""

import json
import sys
import time
from pathlib import Path

from .pressure_profile import validate_kineto


def main():
    request = json.load(sys.stdin)
    started = time.monotonic()
    ranks = {}
    for rank, name in request["paths"].items():
        before = time.monotonic()
        path = Path(name)
        try:
            with path.open() as stream:
                trace = json.load(stream)
            result = validate_kineto(
                trace, request["steps"].get(rank, []), request["requested_steps"]
            )
            del trace
            result["artifact_bytes"] = path.stat().st_size
        except (OSError, ValueError) as exc:
            result = dict(
                trace_valid=False, reasons=["invalid_trace_file"], error=repr(exc)
            )
        result["analysis_elapsed_s"] = time.monotonic() - before
        ranks[rank] = result
    json.dump(
        dict(ranks=ranks, reasons=[], elapsed_s=time.monotonic() - started), sys.stdout
    )


if __name__ == "__main__":
    main()
