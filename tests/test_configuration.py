from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from llm_benchmark.configuration import ConfigurationError, load_generation_settings


class GenerationConfigurationTests(unittest.TestCase):
    def _write(self, directory: str, content: str) -> Path:
        path = Path(directory) / "config.toml"
        path.write_text(content, encoding="utf-8")
        return path

    def test_loads_generation_settings(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = self._write(
                directory,
                """[generation]
enable_thinking = false
max_new_tokens = 256
temperature = 0.2
top_p = 0.8
top_k = 10
do_sample = false
repetition_penalty = 1.1
""",
            )
            settings = load_generation_settings(path)

        self.assertFalse(settings.enable_thinking)
        self.assertEqual(settings.max_new_tokens, 256)
        self.assertEqual(settings.temperature, 0.2)
        self.assertFalse(settings.do_sample)

    def test_missing_values_keep_application_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = self._write(directory, "[generation]\nmax_new_tokens = 64\n")
            settings = load_generation_settings(path)

        self.assertEqual(settings.max_new_tokens, 64)
        self.assertTrue(settings.enable_thinking)
        self.assertEqual(settings.top_p, 0.95)

    def test_unknown_setting_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = self._write(directory, "[generation]\nunknown = 1\n")
            with self.assertRaisesRegex(ConfigurationError, "unknown"):
                load_generation_settings(path)

    def test_invalid_type_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = self._write(
                directory, '[generation]\nmax_new_tokens = "many"\n'
            )
            with self.assertRaisesRegex(ConfigurationError, "must be an integer"):
                load_generation_settings(path)

    def test_missing_file_is_clear(self) -> None:
        with self.assertRaisesRegex(ConfigurationError, "not found"):
            load_generation_settings(Path("does-not-exist.toml"))


if __name__ == "__main__":
    unittest.main()
