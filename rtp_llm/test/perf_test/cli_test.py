import unittest

from rtp_llm.test.perf_test.cli import CliRequest, parse_cli
from rtp_llm.test.perf_test.errors import ConfigurationError


class CliParsingTest(unittest.TestCase):
    def test_run_accepts_only_config_source(self):
        request = parse_cli(
            [
                "--mode",
                "prefill",
                "--action",
                "run",
                "--config",
                "configs/prefill.json",
                "--dry-run",
            ]
        )

        self.assertEqual(
            CliRequest(
                mode="prefill",
                action="run",
                config="configs/prefill.json",
                run_dir=None,
                cases=(),
                dry_run=True,
            ),
            request,
        )

    def test_persisted_actions_accept_only_run_directory(self):
        for action in ("resume", "report"):
            with self.subTest(action=action):
                request = parse_cli(
                    [
                        "--mode",
                        "prefill",
                        "--action",
                        action,
                        "--run-dir",
                        "runs/example",
                    ]
                )
                self.assertEqual("runs/example", request.run_dir)
                self.assertIsNone(request.config)

        profile = parse_cli(
            [
                "--mode",
                "prefill",
                "--action",
                "profile",
                "--run-dir",
                "runs/example",
                "--cases",
                "case-a",
                "case-b",
            ]
        )
        self.assertEqual(("case-a", "case-b"), profile.cases)

    def test_action_source_conflicts_are_exit_code_two(self):
        invalid = [
            ["--mode", "prefill", "--action", "run"],
            [
                "--mode",
                "prefill",
                "--action",
                "run",
                "--config",
                "config.json",
                "--run-dir",
                "runs/example",
            ],
            ["--mode", "prefill", "--action", "resume", "--config", "config.json"],
            [
                "--mode",
                "prefill",
                "--action",
                "report",
                "--run-dir",
                "runs/example",
                "--config",
                "config.json",
            ],
        ]
        for argv in invalid:
            with self.subTest(argv=argv):
                with self.assertRaises(ConfigurationError) as context:
                    parse_cli(argv)
                self.assertEqual(2, context.exception.exit_code)

    def test_persisted_run_overrides_and_removed_flags_are_rejected(self):
        invalid = [
            [
                "--mode",
                "prefill",
                "--action",
                "resume",
                "--run-dir",
                "runs/example",
                "--cases",
                "case-a",
            ],
            [
                "--mode",
                "prefill",
                "--action",
                "resume",
                "--run-dir",
                "runs/example",
                "--engine-arg",
                "reuse_cache=1",
            ],
            [
                "--mode",
                "decode",
                "--action",
                "run",
                "--config",
                "config.json",
                "--partial",
                "1",
            ],
            [
                "--mode",
                "prefill",
                "--action",
                "run",
                "--config",
                "config.json",
                "--cache_grid_json",
                "grid.json",
            ],
            [
                "--mode",
                "prefill",
                "--action",
                "run",
                "--config",
                "config.json",
                "--checkpoint_path",
                "/models/legacy",
            ],
            [
                "--mode",
                "decode",
                "--action",
                "run",
                "--config",
                "config.json",
                "--reuse_cache=1",
            ],
        ]
        for argv in invalid:
            with self.subTest(argv=argv):
                with self.assertRaises(ConfigurationError) as context:
                    parse_cli(argv)
                self.assertEqual(2, context.exception.exit_code)

    def test_required_and_choice_errors_are_typed(self):
        for argv in (
            [],
            ["--mode", "invalid", "--action", "run", "--config", "config.json"],
            ["--mode", "prefill", "--action", "invalid", "--config", "config.json"],
        ):
            with self.subTest(argv=argv):
                with self.assertRaises(ConfigurationError) as context:
                    parse_cli(argv)
                self.assertEqual(2, context.exception.exit_code)

    def test_duplicate_or_empty_profile_case_ids_are_rejected(self):
        for cases in (("case-a", "case-a"), ("",)):
            with self.subTest(cases=cases):
                argv = [
                    "--mode",
                    "prefill",
                    "--action",
                    "profile",
                    "--run-dir",
                    "runs/example",
                    "--cases",
                    *cases,
                ]
                with self.assertRaises(ConfigurationError):
                    parse_cli(argv)


if __name__ == "__main__":
    unittest.main()
