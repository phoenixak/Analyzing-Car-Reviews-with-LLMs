"""
Tests for the error handling system.
"""

import pytest
from unittest.mock import patch, Mock

from src.error_handler import (
    ErrorHandler,
    CarReviewsError,
    ModelLoadError,
    DataProcessingError,
    InsufficientResourcesError,
    ErrorSeverity,
    robust_operation,
    validate_system_resources,
    safe_model_load,
    safe_data_processing,
)


class TestErrorHandler:
    """Test cases for the ErrorHandler class."""

    def test_error_handler_creation(self):
        """Test error handler initializes with empty dicts."""
        handler = ErrorHandler()
        assert isinstance(handler.error_counts, dict)
        assert isinstance(handler.fallback_strategies, dict)

    def test_register_fallback(self):
        """Test registering a fallback strategy for a given exception type."""
        handler = ErrorHandler()

        def mock_fallback(error, context):
            return "fallback_result"

        handler.register_fallback(ValueError, mock_fallback)
        assert ValueError in handler.fallback_strategies

    def test_handle_error_with_fallback(self):
        """Test that handle_error invokes the registered fallback."""
        handler = ErrorHandler()

        def mock_fallback(error, context):
            return "fallback_executed"

        handler.register_fallback(ValueError, mock_fallback)

        result = handler.handle_error(ValueError("test error"), "test_context")
        assert result == "fallback_executed"

    def test_handle_error_without_fallback(self):
        """Test that handle_error returns None when no fallback is registered."""
        handler = ErrorHandler()
        result = handler.handle_error(RuntimeError("test error"), "test_context")
        assert result is None

    def test_error_counting(self):
        """Test that handle_error increments the error count correctly."""
        handler = ErrorHandler()

        handler.handle_error(ValueError("test error"), "test_context")
        handler.handle_error(ValueError("test error"), "test_context")

        error_key = "test_context:ValueError"
        assert handler.error_counts[error_key] == 2

    def test_get_error_summary(self):
        """Test that error summary contains expected keys and totals."""
        handler = ErrorHandler()
        handler.handle_error(ValueError("test1"), "context1")
        handler.handle_error(RuntimeError("test2"), "context2")

        summary = handler.get_error_summary()

        assert "total_errors" in summary
        assert "error_breakdown" in summary
        assert "registered_fallbacks" in summary
        assert summary["total_errors"] == 2


class TestCarReviewsErrors:
    """Test custom exception classes."""

    def test_car_reviews_error_creation(self):
        """Test CarReviewsError stores all constructor parameters."""
        suggestions = ["suggestion1", "suggestion2"]
        context = {"key": "value"}

        error = CarReviewsError(
            "test message",
            severity=ErrorSeverity.HIGH,
            suggestions=suggestions,
            context=context,
        )

        assert error.message == "test message"
        assert error.severity == ErrorSeverity.HIGH
        assert error.suggestions == suggestions
        assert error.context == context

    def test_car_reviews_error_defaults(self):
        """Test CarReviewsError uses sensible defaults."""
        error = CarReviewsError("msg")
        assert error.severity == ErrorSeverity.MEDIUM
        assert error.suggestions == []
        assert error.context == {}

    def test_model_load_error(self):
        """Test ModelLoadError is a subclass of CarReviewsError."""
        error = ModelLoadError("Model failed to load", severity=ErrorSeverity.CRITICAL)
        assert isinstance(error, CarReviewsError)
        assert error.severity == ErrorSeverity.CRITICAL

    def test_data_processing_error(self):
        """Test DataProcessingError is a subclass of CarReviewsError."""
        error = DataProcessingError("Data processing failed")
        assert isinstance(error, CarReviewsError)

    def test_insufficient_resources_error(self):
        """Test InsufficientResourcesError is a subclass of CarReviewsError."""
        error = InsufficientResourcesError("Not enough memory")
        assert isinstance(error, CarReviewsError)


class TestRobustOperation:
    """Test the robust_operation decorator."""

    def test_successful_operation(self):
        """Test that a successful function returns its result directly."""

        @robust_operation(fallback_value="fallback", max_retries=2)
        def successful_function():
            return "success"

        result = successful_function()
        assert result == "success"

    @patch("src.error_handler.time.sleep")
    def test_operation_with_retries(self, mock_sleep):
        """Test that the decorator retries and eventually succeeds."""
        call_count = 0

        @robust_operation(fallback_value="fallback", max_retries=2)
        def failing_then_succeeding():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("temporary failure")
            return "success"

        result = failing_then_succeeding()
        assert result == "success"
        assert call_count == 3
        assert mock_sleep.call_count == 2

    @patch("src.error_handler.time.sleep")
    def test_operation_with_fallback(self, mock_sleep):
        """Test that the decorator returns fallback_value after all retries fail."""

        @robust_operation(fallback_value="fallback", max_retries=1)
        def always_failing():
            raise ValueError("always fails")

        result = always_failing()
        assert result == "fallback"

    @patch("src.error_handler.time.sleep")
    def test_exponential_backoff(self, mock_sleep):
        """Test that retries use exponential backoff sleep times."""

        @robust_operation(fallback_value=None, max_retries=3)
        def always_fails():
            raise RuntimeError("fail")

        always_fails()

        sleep_calls = [c.args[0] for c in mock_sleep.call_args_list]
        assert sleep_calls == [1, 2, 4]


class TestSystemValidation:
    """Test system resource validation."""

    @patch("src.error_handler.psutil.disk_usage")
    @patch("src.error_handler.psutil.virtual_memory")
    def test_validate_system_resources_sufficient(self, mock_memory, mock_disk):
        """Test validation passes with sufficient resources."""
        mock_memory.return_value = Mock(available=1024 * 1024 * 1024)  # 1 GB
        mock_disk.return_value = Mock(free=1024 * 1024 * 1024)  # 1 GB

        result = validate_system_resources(min_memory_mb=512, min_disk_mb=100)
        assert result is True

    @patch("src.error_handler.psutil.virtual_memory")
    def test_validate_system_resources_insufficient_memory(self, mock_memory):
        """Test validation raises InsufficientResourcesError for low memory."""
        mock_memory.return_value = Mock(available=100 * 1024 * 1024)  # 100 MB

        with pytest.raises(InsufficientResourcesError):
            validate_system_resources(min_memory_mb=512, min_disk_mb=100)

    @patch("src.error_handler.psutil.disk_usage")
    @patch("src.error_handler.psutil.virtual_memory")
    def test_validate_system_resources_insufficient_disk(self, mock_memory, mock_disk):
        """Test validation raises InsufficientResourcesError for low disk space."""
        mock_memory.return_value = Mock(available=1024 * 1024 * 1024)
        mock_disk.return_value = Mock(free=10 * 1024 * 1024)  # 10 MB

        with pytest.raises(InsufficientResourcesError):
            validate_system_resources(min_memory_mb=512, min_disk_mb=100)


class TestSafeModelLoad:
    """Test safe model loading."""

    @patch("src.error_handler.validate_system_resources", return_value=True)
    def test_safe_model_load_success(self, _mock_validate):
        """Test successful model loading returns the loaded model."""

        def mock_loader():
            return "loaded_model"

        result = safe_model_load(mock_loader, "test_model")
        assert result == "loaded_model"

    @patch("src.error_handler.validate_system_resources", return_value=True)
    def test_safe_model_load_import_error(self, _mock_validate):
        """Test that ImportError is wrapped into ModelLoadError."""

        def mock_loader():
            raise ImportError("Module not found")

        with pytest.raises(ModelLoadError):
            safe_model_load(mock_loader, "test_model")

    @patch("src.error_handler.validate_system_resources", return_value=True)
    def test_safe_model_load_general_error(self, _mock_validate):
        """Test that a general exception is wrapped into ModelLoadError."""

        def mock_loader():
            raise RuntimeError("General error")

        with pytest.raises(ModelLoadError):
            safe_model_load(mock_loader, "test_model")


class TestSafeDataProcessing:
    """Test safe data processing."""

    def test_safe_data_processing_success(self):
        """Test successful data processing returns the processed result."""

        def mock_processor(data):
            return data.upper()

        result = safe_data_processing(mock_processor, "test data", "test operation")
        assert result == "TEST DATA"

    def test_safe_data_processing_empty_data(self):
        """Test that None/empty data raises DataProcessingError."""

        def mock_processor(data):
            return data

        with pytest.raises(DataProcessingError):
            safe_data_processing(mock_processor, None, "test operation")

    def test_safe_data_processing_error(self):
        """Test that processor exceptions are wrapped into DataProcessingError."""

        def mock_processor(data):
            raise ValueError("Processing error")

        with pytest.raises(DataProcessingError):
            safe_data_processing(mock_processor, "test data", "test operation")
