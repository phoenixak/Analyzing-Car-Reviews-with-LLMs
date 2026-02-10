"""
Tests for the model cache system.
"""

import time
import threading

import pytest
from unittest.mock import Mock, MagicMock, patch

from src.model_cache import ModelCache, PerformanceOptimizedPipeline


@pytest.fixture(autouse=True)
def _mock_system_deps():
    """Mock torch.cuda and psutil for all tests so no GPU/hardware is required."""
    with (
        patch("src.model_cache.torch.cuda.is_available", return_value=False),
        patch("src.model_cache.torch.cuda.empty_cache"),
        patch("src.model_cache.psutil.virtual_memory", return_value=Mock(percent=50)),
    ):
        yield


@pytest.fixture()
def cache():
    """Provide a clean ModelCache singleton for each test."""
    c = ModelCache()
    c.cache.clear()
    c.access_times.clear()
    c.max_cache_size = 10
    c.memory_threshold = 0.8
    yield c
    c.cache.clear()
    c.access_times.clear()


class TestModelCache:
    """Test cases for the ModelCache class."""

    def test_cache_singleton(self):
        """Test that ModelCache always returns the same instance."""
        cache1 = ModelCache()
        cache2 = ModelCache()
        assert cache1 is cache2

    def test_cache_get_with_loader(self, cache):
        """Test cache get with a loader function populates the cache."""
        result = cache.get("test_key", lambda: "test_model")
        assert result == "test_model"
        assert "test_key" in cache.cache

    def test_cache_get_existing(self, cache):
        """Test cache get returns the existing cached item."""
        cache.cache["test_key"] = "cached_model"
        cache.access_times["test_key"] = time.time()

        result = cache.get("test_key")
        assert result == "cached_model"

    def test_cache_get_without_loader_raises(self, cache):
        """Test cache get raises ValueError when key not found and no loader given."""
        with pytest.raises(ValueError):
            cache.get("non_existing_key")

    def test_cache_cleanup_memory_threshold(self, cache):
        """Test cleanup removes items when memory usage exceeds threshold."""
        # Fill cache
        for i in range(5):
            cache.cache[f"key_{i}"] = f"model_{i}"
            cache.access_times[f"key_{i}"] = time.time() - i

        initial_count = len(cache.cache)

        # Override the autouse psutil mock to report 90% memory
        with patch(
            "src.model_cache.psutil.virtual_memory",
            return_value=Mock(percent=90),
        ):
            cache.cleanup_cache()

        assert len(cache.cache) < initial_count

    def test_cache_cleanup_size_threshold(self, cache):
        """Test cleanup removes items when cache exceeds max_cache_size."""
        cache.max_cache_size = 3
        for i in range(5):
            cache.cache[f"key_{i}"] = f"model_{i}"
            cache.access_times[f"key_{i}"] = time.time() - i

        cache.cleanup_cache()
        assert len(cache.cache) <= cache.max_cache_size

    def test_get_memory_usage(self, cache):
        """Test get_memory_usage returns fraction of system memory used."""
        with patch(
            "src.model_cache.psutil.virtual_memory",
            return_value=Mock(percent=75),
        ):
            usage = cache.get_memory_usage()
        assert usage == 0.75

    def test_clear_cache(self, cache):
        """Test clear_cache empties all cache data structures."""
        cache.cache["test_key"] = "test_model"
        cache.access_times["test_key"] = time.time()

        cache.clear_cache()

        assert len(cache.cache) == 0
        assert len(cache.access_times) == 0

    def test_get_cache_info(self, cache):
        """Test get_cache_info returns expected keys and values."""
        cache.cache["test_key"] = "test_model"

        info = cache.get_cache_info()

        assert "cached_models" in info
        assert "cache_size" in info
        assert "memory_usage" in info
        assert "max_cache_size" in info
        assert "memory_threshold" in info
        assert info["cache_size"] == 1
        assert "test_key" in info["cached_models"]

    @patch("src.model_cache.AutoTokenizer.from_pretrained")
    @patch("src.model_cache.pipeline")
    def test_preload_models(self, mock_pipeline_fn, mock_tokenizer, cache):
        """Test preload_models loads models without raising."""
        mock_pipeline_fn.return_value = MagicMock()
        mock_tokenizer.return_value = MagicMock()

        cache.preload_models()

        assert mock_pipeline_fn.call_count >= 3
        assert mock_tokenizer.call_count >= 1


class TestPerformanceOptimizedPipeline:
    """Test cases for PerformanceOptimizedPipeline."""

    def test_pipeline_initialization(self, cache):
        """Test that pipeline stores cache_key and references ModelCache."""
        pipe = PerformanceOptimizedPipeline("test_cache_key")
        assert pipe.cache_key == "test_cache_key"
        assert isinstance(pipe.cache, ModelCache)

    @patch.object(ModelCache, "get")
    def test_get_cached_model(self, mock_get, cache):
        """Test get_cached_model delegates to ModelCache.get."""
        mock_model = Mock()
        mock_get.return_value = mock_model

        pipe = PerformanceOptimizedPipeline("test_key")
        result = pipe.get_cached_model(lambda: "loaded_model")

        mock_get.assert_called_once()
        assert result == mock_model


class TestOptimizedPipelines:
    """Test cases for the optimized pipeline subclasses."""

    @patch("src.model_cache.pipeline")
    def test_optimized_sentiment_pipeline(self, mock_pipeline_fn, cache):
        """Test OptimizedSentimentPipeline calls the cached classifier."""
        from src.model_cache import OptimizedSentimentPipeline

        mock_classifier = Mock()
        mock_classifier.return_value = [{"label": "POSITIVE", "score": 0.9}]
        mock_pipeline_fn.return_value = mock_classifier

        pipe = OptimizedSentimentPipeline()
        result = pipe("Great car!")

        mock_classifier.assert_called_once_with("Great car!")
        assert result == [{"label": "POSITIVE", "score": 0.9}]

    @patch("src.model_cache.pipeline")
    def test_optimized_translation_pipeline(self, mock_pipeline_fn, cache):
        """Test OptimizedTranslationPipeline extracts translation_text."""
        from src.model_cache import OptimizedTranslationPipeline

        mock_translator = Mock()
        mock_translator.return_value = [{"translation_text": "Excelente coche!"}]
        mock_pipeline_fn.return_value = mock_translator

        pipe = OptimizedTranslationPipeline()
        result = pipe("Great car!")

        mock_translator.assert_called_once()
        assert result == "Excelente coche!"

    @patch("src.model_cache.pipeline")
    def test_optimized_summarization_pipeline(self, mock_pipeline_fn, cache):
        """Test OptimizedSummarizationPipeline extracts summary_text."""
        from src.model_cache import OptimizedSummarizationPipeline

        mock_summarizer = Mock()
        mock_summarizer.return_value = [{"summary_text": "Great car with excellent features."}]
        mock_pipeline_fn.return_value = mock_summarizer

        pipe = OptimizedSummarizationPipeline()
        long_text = "This is a great car with excellent features. " * 10
        result = pipe(long_text)

        mock_summarizer.assert_called_once()
        assert result == "Great car with excellent features."


class TestCachePerformance:
    """Performance tests for the cache system."""

    def test_cache_performance_improvement(self, cache):
        """Test that cache hits are faster than initial loads."""
        call_count = 0

        def slow_loader():
            nonlocal call_count
            call_count += 1
            time.sleep(0.05)
            return f"model_{call_count}"

        # First call loads
        result1 = cache.get("perf_test", slow_loader)

        # Second call from cache
        start = time.time()
        result2 = cache.get("perf_test")
        cache_time = time.time() - start

        assert result1 == result2
        assert call_count == 1
        assert cache_time < 0.01

    def test_multiple_model_caching(self, cache):
        """Test caching and retrieving multiple models."""
        models = ["sentiment", "translation", "summarization"]

        for name in models:
            cache.get(f"{name}_model", lambda n=name: f"loaded_{n}")

        for name in models:
            result = cache.get(f"{name}_model")
            assert result == f"loaded_{name}"

        assert len(cache.cache) == len(models)


class TestCacheEdgeCases:
    """Test edge cases for the cache system."""

    def test_cache_with_exception_in_loader(self, cache):
        """Test that a loader exception does not leave a corrupted cache entry."""

        def failing_loader():
            raise RuntimeError("Loader failed")

        with pytest.raises(RuntimeError):
            cache.get("failing_key", failing_loader)

        assert "failing_key" not in cache.cache

    def test_cache_access_time_updates(self, cache):
        """Test that accessing a cached item updates its access time."""
        cache.cache["test_key"] = "test_model"
        old_time = time.time() - 100
        cache.access_times["test_key"] = old_time

        time.sleep(0.01)
        cache.get("test_key")

        assert cache.access_times["test_key"] > old_time

    def test_cache_thread_safety(self, cache):
        """Test that concurrent cache access from multiple threads works."""
        results = {}

        def cache_operation(thread_id):
            def loader():
                time.sleep(0.01)
                return f"model_for_thread_{thread_id}"

            results[thread_id] = cache.get(f"thread_key_{thread_id}", loader)

        threads = [threading.Thread(target=cache_operation, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(results) == 5
        for i in range(5):
            assert results[i] == f"model_for_thread_{i}"
