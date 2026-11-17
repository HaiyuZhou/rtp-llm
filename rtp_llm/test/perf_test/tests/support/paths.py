"""Locations of production fixtures, independent of test module depth."""

from pathlib import Path

PERF_TEST_ROOT = Path(__file__).resolve().parents[2]
