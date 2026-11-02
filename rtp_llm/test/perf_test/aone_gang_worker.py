"""Bazel entry point for one generated gang shard target."""

from rtp_llm.test.perf_test.aone.gang_worker import main

if __name__ == "__main__":
    raise SystemExit(main())
