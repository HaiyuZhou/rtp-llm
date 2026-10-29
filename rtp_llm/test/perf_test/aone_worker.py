"""Bazel entry point for one generated AOne shard target."""

from rtp_llm.test.perf_test.aone.worker import main

if __name__ == "__main__":
    raise SystemExit(main())
