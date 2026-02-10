"""Tests for the enhanced CLI module (src.enhanced_cli).

Covers the EnhancedCLI class methods and the enhanced_main Click command.
All model / pipeline calls are mocked so tests run without GPU or downloads.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from src.enhanced_cli import EnhancedCLI, enhanced_main


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def runner():
    """Click CLI test runner."""
    return CliRunner()


@pytest.fixture()
def cli():
    """Fresh EnhancedCLI instance."""
    return EnhancedCLI()


@pytest.fixture()
def temp_csv(tmp_path):
    """Create a minimal CSV file accepted by --data-file."""
    path = tmp_path / "reviews.csv"
    path.write_text(
        "Review;Class\nGreat car!;positive\nBad car.;negative\n",
        encoding="utf-8",
    )
    return str(path)


# ---------------------------------------------------------------------------
# EnhancedCLI class
# ---------------------------------------------------------------------------


class TestEnhancedCLIClass:
    """Unit tests for EnhancedCLI methods."""

    def test_init_attributes(self, cli):
        """Constructor sets console, progress, and current_tasks."""
        assert cli.console is not None
        assert cli.progress is None
        assert isinstance(cli.current_tasks, dict)

    def test_validate_inputs_valid(self, cli, temp_csv):
        """Valid file + valid task returns True."""
        assert cli.validate_inputs(temp_csv, ["sentiment"]) is True

    def test_validate_inputs_all_tasks(self, cli, temp_csv):
        """Every recognised task name passes validation."""
        valid = [
            "sentiment",
            "translation",
            "qa",
            "summarization",
            "topic",
            "aspect",
            "ner",
            "all",
        ]
        for task in valid:
            assert cli.validate_inputs(temp_csv, [task]) is True

    def test_validate_inputs_bad_file(self, cli):
        """Non-existent file causes validation to return False."""
        assert cli.validate_inputs("/no/such/file.csv", ["sentiment"]) is False

    def test_validate_inputs_bad_task(self, cli, temp_csv):
        """Unrecognised task name causes validation to return False."""
        assert cli.validate_inputs(temp_csv, ["bogus_task"]) is False

    def test_create_progress_tracker(self, cli):
        """create_progress_tracker returns and stores a Progress object."""
        progress = cli.create_progress_tracker("Test")
        assert progress is not None
        assert cli.progress is progress

    def test_display_banner_does_not_raise(self, cli):
        """display_banner should execute without errors."""
        cli.display_banner()

    def test_display_help_examples_does_not_raise(self, cli):
        """display_help_examples should execute without errors."""
        cli.display_help_examples()

    def test_display_results_summary(self, cli):
        """display_results_summary handles typical result dict."""
        results = {
            "sentiment": {
                "metrics": {"accuracy": 0.85},
                "predictions": [{"label": "POSITIVE", "score": 0.9}],
            }
        }
        cli.display_results_summary(results)  # should not raise

    def test_display_model_info(self, cli):
        """display_model_info should execute without errors."""
        cli.display_model_info()


# ---------------------------------------------------------------------------
# enhanced_main Click command
# ---------------------------------------------------------------------------


class TestEnhancedMainCommand:
    """Tests for the enhanced_main Click entry point."""

    def test_help_flag(self, runner):
        """--help exits 0 and mentions key options."""
        result = runner.invoke(enhanced_main, ["--help"])
        assert result.exit_code == 0
        assert "--task" in result.output
        assert "--visualize" in result.output
        assert "--save-results" in result.output
        assert "--show-examples" in result.output
        assert "--show-models" in result.output
        assert "--show-config" in result.output
        assert "--preload-models" in result.output

    def test_show_examples(self, runner, temp_csv):
        """--show-examples displays examples and exits successfully."""
        result = runner.invoke(
            enhanced_main,
            ["--data-file", temp_csv, "--show-examples"],
        )
        assert result.exit_code == 0

    def test_show_models(self, runner, temp_csv):
        """--show-models displays model info and exits successfully."""
        result = runner.invoke(
            enhanced_main,
            ["--data-file", temp_csv, "--show-models"],
        )
        assert result.exit_code == 0

    def test_show_config(self, runner, temp_csv):
        """--show-config displays system info and config, then exits."""
        result = runner.invoke(
            enhanced_main,
            ["--data-file", temp_csv, "--show-config"],
        )
        assert result.exit_code == 0

    @patch("src.enhanced_cli.model_cache")
    def test_preload_models(self, mock_cache, runner, temp_csv):
        """--preload-models calls model_cache.preload_models()."""
        mock_cache.preload_models = MagicMock()
        result = runner.invoke(
            enhanced_main,
            ["--data-file", temp_csv, "--preload-models"],
        )
        assert result.exit_code == 0
        mock_cache.preload_models.assert_called_once()

    def test_nonexistent_data_file(self, runner):
        """Passing a path that doesn't exist is rejected by Click."""
        result = runner.invoke(
            enhanced_main,
            ["--data-file", "/no/such/file.csv", "--task", "sentiment"],
        )
        assert result.exit_code != 0
