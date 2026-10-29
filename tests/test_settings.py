import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from pydantic import ValidationError

from pyshell.settings import Settings


class SettingsTests(unittest.TestCase):
    def setUp(self):
        self.environment = patch.dict(os.environ, {}, clear=True)
        self.environment.start()
        self.addCleanup(self.environment.stop)

    def test_loads_dotenv_from_current_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, ".env").write_text(
                "OPENAI_API_KEY=test-key\nOPENAI_MODEL=test-model\n",
                encoding="utf-8",
            )
            previous = Path.cwd()
            try:
                os.chdir(directory)
                settings = Settings()
            finally:
                os.chdir(previous)
        self.assertEqual(settings.openai_api_key.get_secret_value(), "test-key")
        self.assertNotIn("test-key", repr(settings))
        self.assertEqual(settings.openai_model, "test-model")

    def test_environment_overrides_dotenv(self):
        with tempfile.TemporaryDirectory() as directory:
            dotenv = Path(directory, ".env")
            dotenv.write_text("OPENAI_API_KEY=file-key\nOPENAI_MODEL=file-model\n")
            os.environ["OPENAI_MODEL"] = "environment-model"
            self.assertEqual(Settings(_env_file=dotenv).openai_model, "environment-model")

    def test_required_key(self):
        for values in ({}, {"openai_api_key": ""}):
            with self.assertRaises(ValidationError):
                Settings(_env_file=None, **values)


if __name__ == "__main__":
    unittest.main()
