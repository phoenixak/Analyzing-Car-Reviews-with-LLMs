"""
Advanced model cache manager for optimized performance.

This module provides centralized model caching to avoid repeated loading
and dramatically improve performance across multiple runs.
"""

import gc
import os
import psutil
import time
from typing import Dict, Any, Optional, Callable
from threading import Lock
import torch
from transformers import pipeline, AutoTokenizer, AutoModel

from src.logger import get_logger
from src.config import SENTIMENT_MODEL, TRANSLATION_MODEL, SUMMARIZATION_MODEL, QA_MODEL

# Set up logger
logger = get_logger(__name__)


class ModelCache:
    """Thread-safe singleton model cache with memory management."""

    _instance = None
    _lock = Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance.cache = {}
                    cls._instance.access_times = {}
                    cls._instance.memory_threshold = 0.8  # 80% memory usage threshold
                    cls._instance.max_cache_size = 10
                    logger.info("Initialized ModelCache singleton")
        return cls._instance

    def get_memory_usage(self) -> float:
        """Get current memory usage percentage."""
        return psutil.virtual_memory().percent / 100.0

    def cleanup_cache(self):
        """Remove least recently used models if memory usage is high."""
        if (
            self.get_memory_usage() > self.memory_threshold
            or len(self.cache) > self.max_cache_size
        ):
            logger.info(
                f"Memory usage: {self.get_memory_usage():.1%}, cleaning cache..."
            )

            # Sort by access time and remove oldest
            sorted_items = sorted(self.access_times.items(), key=lambda x: x[1])
            items_to_remove = len(sorted_items) // 2  # Remove half the cache

            for model_key, _ in sorted_items[:items_to_remove]:
                if model_key in self.cache:
                    del self.cache[model_key]
                    del self.access_times[model_key]
                    logger.info(f"Removed {model_key} from cache")

            # Force garbage collection
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def get(
        self, key: str, loader_func: Optional[Callable] = None, *args, **kwargs
    ) -> Any:
        """
        Get model from cache or load it.

        Args:
            key: Unique model identifier
            loader_func: Function to load the model if not cached
            *args, **kwargs: Arguments for loader function

        Returns:
            Cached or newly loaded model
        """
        with self._lock:
            # Update access time
            self.access_times[key] = time.time()

            # Return cached model if available
            if key in self.cache:
                logger.debug(f"Retrieved {key} from cache")
                return self.cache[key]

            # Check memory before loading new model
            self.cleanup_cache()

            # Load new model
            if loader_func:
                logger.info(f"Loading new model: {key}")
                start_time = time.time()
                model = loader_func(*args, **kwargs)
                load_time = time.time() - start_time

                self.cache[key] = model
                self.access_times[key] = time.time()

                logger.info(
                    f"Loaded {key} in {load_time:.2f}s, cache size: {len(self.cache)}"
                )
                return model

            raise ValueError(
                f"Model {key} not in cache and no loader function provided"
            )

    def preload_models(self):
        """Preload commonly used models for better performance."""
        logger.info("Preloading commonly used models...")

        try:
            # Preload sentiment analysis
            self.get(
                "sentiment_pipeline",
                lambda: pipeline("sentiment-analysis", model=SENTIMENT_MODEL),
            )

            # Preload translation
            self.get(
                "translation_pipeline",
                lambda: pipeline("translation", model=TRANSLATION_MODEL),
            )

            # Preload summarization
            self.get(
                "summarization_pipeline",
                lambda: pipeline("summarization", model=SUMMARIZATION_MODEL),
            )

            # Preload QA model tokenizer and model
            self.get("qa_tokenizer", lambda: AutoTokenizer.from_pretrained(QA_MODEL))

            logger.info(f"Preloaded {len(self.cache)} models successfully")

        except Exception as e:
            logger.warning(f"Failed to preload some models: {e}")

    def clear_cache(self):
        """Clear all cached models."""
        with self._lock:
            self.cache.clear()
            self.access_times.clear()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info("Cleared model cache")

    def get_cache_info(self) -> Dict[str, Any]:
        """Get information about current cache state."""
        return {
            "cached_models": list(self.cache.keys()),
            "cache_size": len(self.cache),
            "memory_usage": f"{self.get_memory_usage():.1%}",
            "max_cache_size": self.max_cache_size,
            "memory_threshold": f"{self.memory_threshold:.1%}",
        }


class PerformanceOptimizedPipeline:
    """Base class for performance-optimized ML pipelines."""

    def __init__(self, cache_key: str):
        self.cache_key = cache_key
        self.cache = ModelCache()
        self.logger = get_logger(self.__class__.__name__)

    def get_cached_model(self, loader_func: Callable, *args, **kwargs):
        """Get model from cache with performance monitoring."""
        start_time = time.time()
        model = self.cache.get(self.cache_key, loader_func, *args, **kwargs)
        load_time = time.time() - start_time

        if load_time > 0.1:  # Only log if loading took significant time
            self.logger.debug(f"Model retrieval took {load_time:.3f}s")

        return model


# Optimized pipeline classes
class OptimizedSentimentPipeline(PerformanceOptimizedPipeline):
    """Memory-optimized sentiment analysis pipeline."""

    def __init__(self, model_name: str = SENTIMENT_MODEL):
        super().__init__(f"sentiment_{model_name}")
        self.model_name = model_name

    def __call__(self, texts):
        classifier = self.get_cached_model(
            lambda: pipeline("sentiment-analysis", model=self.model_name)
        )
        return classifier(texts)


class OptimizedTranslationPipeline(PerformanceOptimizedPipeline):
    """Memory-optimized translation pipeline."""

    def __init__(self, model_name: str = TRANSLATION_MODEL):
        super().__init__(f"translation_{model_name}")
        self.model_name = model_name

    def __call__(self, text, max_length: int = 512):
        translator = self.get_cached_model(
            lambda: pipeline("translation", model=self.model_name)
        )
        result = translator(text, max_length=max_length)
        return result[0]["translation_text"]


class OptimizedSummarizationPipeline(PerformanceOptimizedPipeline):
    """Memory-optimized summarization pipeline."""

    def __init__(self, model_name: str = SUMMARIZATION_MODEL):
        super().__init__(f"summarization_{model_name}")
        self.model_name = model_name

    def __call__(self, text, max_length: int = 100, min_length: int = 30):
        summarizer = self.get_cached_model(
            lambda: pipeline("summarization", model=self.model_name)
        )
        result = summarizer(text, max_length=max_length, min_length=min_length)
        return result[0]["summary_text"]


# Global cache instance
model_cache = ModelCache()
