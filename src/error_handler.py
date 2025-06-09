"""
Comprehensive error handling and recovery system.

This module provides robust error handling, graceful degradation,
and user-friendly error messages across the application.
"""

import traceback
import functools
from typing import Any, Callable, Dict, List, Optional, Union
from enum import Enum
import psutil
import sys

from src.logger import get_logger

logger = get_logger(__name__)

class ErrorSeverity(Enum):
    """Error severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class CarReviewsError(Exception):
    """Base exception for car reviews analysis application."""
    
    def __init__(self, message: str, severity: ErrorSeverity = ErrorSeverity.MEDIUM, 
                 suggestions: Optional[List[str]] = None, context: Optional[Dict] = None):
        self.message = message
        self.severity = severity
        self.suggestions = suggestions or []
        self.context = context or {}
        super().__init__(message)

class ModelLoadError(CarReviewsError):
    """Exception raised when model loading fails."""
    pass

class DataProcessingError(CarReviewsError):
    """Exception raised when data processing fails."""
    pass

class VisualizationError(CarReviewsError):
    """Exception raised when visualization creation fails."""
    pass

class InsufficientResourcesError(CarReviewsError):
    """Exception raised when system resources are insufficient."""
    pass

class ErrorHandler:
    """Centralized error handling and recovery system."""
    
    def __init__(self):
        self.error_counts = {}
        self.fallback_strategies = {}
        self.recovery_attempts = {}
        
    def register_fallback(self, error_type: type, fallback_func: Callable):
        """Register a fallback strategy for a specific error type."""
        self.fallback_strategies[error_type] = fallback_func
        logger.info(f"Registered fallback for {error_type.__name__}")
    
    def handle_error(self, error: Exception, context: str = "Unknown") -> Optional[Any]:
        """
        Handle an error with appropriate recovery strategy.
        
        Args:
            error: The exception that occurred
            context: Context where the error occurred
            
        Returns:
            Result from fallback strategy if available, None otherwise
        """
        error_type = type(error)
        error_key = f"{context}:{error_type.__name__}"
        
        # Increment error count
        self.error_counts[error_key] = self.error_counts.get(error_key, 0) + 1
        
        # Log the error with appropriate level
        severity = getattr(error, 'severity', ErrorSeverity.MEDIUM)
        log_level = {
            ErrorSeverity.LOW: "warning",
            ErrorSeverity.MEDIUM: "error", 
            ErrorSeverity.HIGH: "error",
            ErrorSeverity.CRITICAL: "critical"
        }[severity]
        
        getattr(logger, log_level)(
            f"Error in {context}: {str(error)}\n"
            f"Error count for {error_key}: {self.error_counts[error_key]}\n"
            f"Traceback: {traceback.format_exc()}"
        )
        
        # Try fallback strategy
        if error_type in self.fallback_strategies:
            try:
                logger.info(f"Attempting fallback strategy for {error_type.__name__}")
                return self.fallback_strategies[error_type](error, context)
            except Exception as fallback_error:
                logger.error(f"Fallback strategy failed: {fallback_error}")
        
        # Provide user-friendly suggestions
        suggestions = getattr(error, 'suggestions', [])
        if suggestions:
            logger.info(f"Suggestions: {'; '.join(suggestions)}")
        
        return None
    
    def get_error_summary(self) -> Dict[str, Any]:
        """Get summary of all errors encountered."""
        return {
            "total_errors": sum(self.error_counts.values()),
            "error_breakdown": dict(self.error_counts),
            "registered_fallbacks": list(self.fallback_strategies.keys())
        }

# Global error handler instance
error_handler = ErrorHandler()

def robust_operation(fallback_value: Any = None, 
                    max_retries: int = 3,
                    context: str = "operation"):
    """
    Decorator for robust operation execution with automatic retries and fallbacks.
    
    Args:
        fallback_value: Value to return if operation fails
        max_retries: Maximum number of retry attempts
        context: Context description for error logging
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    
                    if attempt < max_retries:
                        wait_time = 2 ** attempt  # Exponential backoff
                        logger.warning(
                            f"Attempt {attempt + 1} failed for {func.__name__}: {e}. "
                            f"Retrying in {wait_time}s..."
                        )
                        import time
                        time.sleep(wait_time)
                    else:
                        logger.error(f"All {max_retries + 1} attempts failed for {func.__name__}")
            
            # Handle the final error
            result = error_handler.handle_error(last_exception, f"{context}:{func.__name__}")
            return result if result is not None else fallback_value
        
        return wrapper
    return decorator

def validate_system_resources(min_memory_mb: int = 512, 
                             min_disk_mb: int = 100) -> bool:
    """
    Validate that system has sufficient resources.
    
    Args:
        min_memory_mb: Minimum required memory in MB
        min_disk_mb: Minimum required disk space in MB
        
    Returns:
        True if resources are sufficient
        
    Raises:
        InsufficientResourcesError: If resources are insufficient
    """
    try:
        # Check memory
        memory = psutil.virtual_memory()
        available_memory_mb = memory.available / (1024 * 1024)
        
        if available_memory_mb < min_memory_mb:
            suggestions = [
                "Close unnecessary applications",
                "Increase system memory",
                "Use smaller batch sizes",
                "Enable model caching to reduce memory usage"
            ]
            raise InsufficientResourcesError(
                f"Insufficient memory: {available_memory_mb:.0f}MB available, "
                f"{min_memory_mb}MB required",
                severity=ErrorSeverity.HIGH,
                suggestions=suggestions,
                context={"available_memory_mb": available_memory_mb, "required_memory_mb": min_memory_mb}
            )
        
        # Check disk space
        disk = psutil.disk_usage('/')
        available_disk_mb = disk.free / (1024 * 1024)
        
        if available_disk_mb < min_disk_mb:
            suggestions = [
                "Free up disk space",
                "Clean temporary files",
                "Use external storage for results"
            ]
            raise InsufficientResourcesError(
                f"Insufficient disk space: {available_disk_mb:.0f}MB available, "
                f"{min_disk_mb}MB required",
                severity=ErrorSeverity.HIGH,
                suggestions=suggestions,
                context={"available_disk_mb": available_disk_mb, "required_disk_mb": min_disk_mb}
            )
        
        logger.info(f"System resources validated: {available_memory_mb:.0f}MB memory, {available_disk_mb:.0f}MB disk")
        return True
        
    except Exception as e:
        logger.error(f"Failed to validate system resources: {e}")
        return False

def safe_model_load(model_loader: Callable, model_name: str, **kwargs) -> Any:
    """
    Safely load a model with comprehensive error handling.
    
    Args:
        model_loader: Function to load the model
        model_name: Name of the model being loaded
        **kwargs: Additional arguments for model loader
        
    Returns:
        Loaded model or None if loading fails
    """
    try:
        # Validate resources before loading
        validate_system_resources(min_memory_mb=256)
        
        logger.info(f"Loading model: {model_name}")
        model = model_loader(**kwargs)
        logger.info(f"Successfully loaded model: {model_name}")
        return model
        
    except ImportError as e:
        suggestions = [
            f"Install missing dependency: pip install {model_name.split('/')[0]}",
            "Check requirements.txt for all dependencies",
            "Use virtual environment to avoid conflicts"
        ]
        raise ModelLoadError(
            f"Missing dependency for {model_name}: {e}",
            severity=ErrorSeverity.HIGH,
            suggestions=suggestions
        )
    
    except Exception as e:
        suggestions = [
            "Check internet connection for model download",
            "Verify model name is correct",
            "Try a smaller model if memory is limited",
            "Clear model cache and retry"
        ]
        raise ModelLoadError(
            f"Failed to load model {model_name}: {e}",
            severity=ErrorSeverity.HIGH,
            suggestions=suggestions,
            context={"model_name": model_name, "error_type": type(e).__name__}
        )

def safe_data_processing(processor: Callable, data: Any, operation: str) -> Any:
    """
    Safely process data with error handling and validation.
    
    Args:
        processor: Function to process the data
        data: Data to process
        operation: Description of the operation
        
    Returns:
        Processed data or None if processing fails
    """
    try:
        if not data:
            raise DataProcessingError(
                f"No data provided for {operation}",
                severity=ErrorSeverity.MEDIUM,
                suggestions=["Check data source", "Verify data loading step"]
            )
        
        result = processor(data)
        
        if not result:
            logger.warning(f"Data processing returned empty result for {operation}")
        
        return result
        
    except Exception as e:
        suggestions = [
            "Check data format and structure",
            "Validate input data quality",
            "Try with smaller data sample",
            "Check for missing values or corrupted data"
        ]
        
        raise DataProcessingError(
            f"Data processing failed for {operation}: {e}",
            severity=ErrorSeverity.MEDIUM,
            suggestions=suggestions,
            context={"operation": operation, "data_type": type(data).__name__}
        )

# Register default fallback strategies
def default_model_fallback(error: Exception, context: str) -> None:
    """Default fallback for model loading errors."""
    logger.warning(f"Using fallback: Skipping model-dependent operation in {context}")
    return None

def default_visualization_fallback(error: Exception, context: str) -> None:
    """Default fallback for visualization errors."""
    logger.warning(f"Using fallback: Skipping visualization in {context}")
    return None

# Register fallbacks
error_handler.register_fallback(ModelLoadError, default_model_fallback)
error_handler.register_fallback(VisualizationError, default_visualization_fallback)