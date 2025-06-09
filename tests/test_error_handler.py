"""
Tests for the error handling system.
"""

import unittest
import sys
from pathlib import Path
from unittest.mock import patch, Mock

# Add the parent directory to the path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.error_handler import (
    ErrorHandler,
    CarReviewsError,
    ModelLoadError,
    DataProcessingError,
    ErrorSeverity,
    robust_operation,
    validate_system_resources,
    safe_model_load,
    safe_data_processing
)


class TestErrorHandler(unittest.TestCase):
    """Test cases for the ErrorHandler class."""
    
    def setUp(self):
        """Set up test error handler."""
        self.error_handler = ErrorHandler()
    
    def test_error_handler_creation(self):
        """Test error handler initialization."""
        self.assertIsInstance(self.error_handler.error_counts, dict)
        self.assertIsInstance(self.error_handler.fallback_strategies, dict)
    
    def test_register_fallback(self):
        """Test registering fallback strategies."""
        def mock_fallback(error, context):
            return "fallback_result"
        
        self.error_handler.register_fallback(ValueError, mock_fallback)
        self.assertIn(ValueError, self.error_handler.fallback_strategies)
    
    def test_handle_error_with_fallback(self):
        """Test error handling with registered fallback."""
        def mock_fallback(error, context):
            return "fallback_executed"
        
        self.error_handler.register_fallback(ValueError, mock_fallback)
        
        test_error = ValueError("test error")
        result = self.error_handler.handle_error(test_error, "test_context")
        
        self.assertEqual(result, "fallback_executed")
    
    def test_handle_error_without_fallback(self):
        """Test error handling without registered fallback."""
        test_error = RuntimeError("test error")
        result = self.error_handler.handle_error(test_error, "test_context")
        
        self.assertIsNone(result)
    
    def test_error_counting(self):
        """Test error counting functionality."""
        test_error = ValueError("test error")
        
        # Handle same error multiple times
        self.error_handler.handle_error(test_error, "test_context")
        self.error_handler.handle_error(test_error, "test_context")
        
        error_key = "test_context:ValueError"
        self.assertEqual(self.error_handler.error_counts[error_key], 2)
    
    def test_get_error_summary(self):
        """Test error summary generation."""
        # Generate some errors
        self.error_handler.handle_error(ValueError("test1"), "context1")
        self.error_handler.handle_error(RuntimeError("test2"), "context2")
        
        summary = self.error_handler.get_error_summary()
        
        self.assertIn("total_errors", summary)
        self.assertIn("error_breakdown", summary)
        self.assertIn("registered_fallbacks", summary)
        self.assertEqual(summary["total_errors"], 2)


class TestCarReviewsErrors(unittest.TestCase):
    """Test custom exception classes."""
    
    def test_car_reviews_error_creation(self):
        """Test CarReviewsError creation with all parameters."""
        suggestions = ["suggestion1", "suggestion2"]
        context = {"key": "value"}
        
        error = CarReviewsError(
            "test message",
            severity=ErrorSeverity.HIGH,
            suggestions=suggestions,
            context=context
        )
        
        self.assertEqual(error.message, "test message")
        self.assertEqual(error.severity, ErrorSeverity.HIGH)
        self.assertEqual(error.suggestions, suggestions)
        self.assertEqual(error.context, context)
    
    def test_model_load_error(self):
        """Test ModelLoadError creation."""
        error = ModelLoadError("Model failed to load", severity=ErrorSeverity.CRITICAL)
        self.assertIsInstance(error, CarReviewsError)
        self.assertEqual(error.severity, ErrorSeverity.CRITICAL)
    
    def test_data_processing_error(self):
        """Test DataProcessingError creation."""
        error = DataProcessingError("Data processing failed")
        self.assertIsInstance(error, CarReviewsError)


class TestRobustOperation(unittest.TestCase):
    """Test the robust_operation decorator."""
    
    def test_successful_operation(self):
        """Test robust operation with successful execution."""
        @robust_operation(fallback_value="fallback", max_retries=2)
        def successful_function():
            return "success"
        
        result = successful_function()
        self.assertEqual(result, "success")
    
    def test_operation_with_retries(self):
        """Test robust operation with retries."""
        call_count = 0
        
        @robust_operation(fallback_value="fallback", max_retries=2)
        def failing_then_succeeding_function():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("temporary failure")
            return "success"
        
        result = failing_then_succeeding_function()
        self.assertEqual(result, "success")
        self.assertEqual(call_count, 3)
    
    def test_operation_with_fallback(self):
        """Test robust operation falling back after all retries."""
        @robust_operation(fallback_value="fallback", max_retries=1)
        def always_failing_function():
            raise ValueError("always fails")
        
        result = always_failing_function()
        self.assertEqual(result, "fallback")


class TestSystemValidation(unittest.TestCase):
    """Test system resource validation."""
    
    @patch('src.error_handler.psutil.virtual_memory')
    @patch('src.error_handler.psutil.disk_usage')
    def test_validate_system_resources_sufficient(self, mock_disk, mock_memory):
        """Test system validation with sufficient resources."""
        # Mock sufficient resources
        mock_memory.return_value = Mock(available=1024 * 1024 * 1024)  # 1GB
        mock_disk.return_value = Mock(free=1024 * 1024 * 1024)  # 1GB
        
        result = validate_system_resources(min_memory_mb=512, min_disk_mb=100)
        self.assertTrue(result)
    
    @patch('src.error_handler.psutil.virtual_memory')
    def test_validate_system_resources_insufficient_memory(self, mock_memory):
        """Test system validation with insufficient memory."""
        # Mock insufficient memory
        mock_memory.return_value = Mock(available=100 * 1024 * 1024)  # 100MB
        
        result = validate_system_resources(min_memory_mb=512, min_disk_mb=100)
        self.assertFalse(result)  # Should return False for insufficient resources


class TestSafeModelLoad(unittest.TestCase):
    """Test safe model loading."""
    
    def test_safe_model_load_success(self):
        """Test successful model loading."""
        def mock_loader():
            return "loaded_model"
        
        result = safe_model_load(mock_loader, "test_model")
        self.assertEqual(result, "loaded_model")
    
    def test_safe_model_load_import_error(self):
        """Test model loading with import error."""
        def mock_loader():
            raise ImportError("Module not found")
        
        with self.assertRaises(ModelLoadError):
            safe_model_load(mock_loader, "test_model")
    
    def test_safe_model_load_general_error(self):
        """Test model loading with general error."""
        def mock_loader():
            raise RuntimeError("General error")
        
        with self.assertRaises(ModelLoadError):
            safe_model_load(mock_loader, "test_model")


class TestSafeDataProcessing(unittest.TestCase):
    """Test safe data processing."""
    
    def test_safe_data_processing_success(self):
        """Test successful data processing."""
        def mock_processor(data):
            return data.upper()
        
        result = safe_data_processing(mock_processor, "test data", "test operation")
        self.assertEqual(result, "TEST DATA")
    
    def test_safe_data_processing_empty_data(self):
        """Test data processing with empty data."""
        def mock_processor(data):
            return data
        
        with self.assertRaises(DataProcessingError):
            safe_data_processing(mock_processor, None, "test operation")
    
    def test_safe_data_processing_error(self):
        """Test data processing with error."""
        def mock_processor(data):
            raise ValueError("Processing error")
        
        with self.assertRaises(DataProcessingError):
            safe_data_processing(mock_processor, "test data", "test operation")


if __name__ == "__main__":
    unittest.main(verbosity=2)