"""
Tests for the model cache system.
"""

import unittest
import sys
import time
from pathlib import Path
from unittest.mock import Mock, patch

# Add the parent directory to the path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.model_cache import ModelCache, PerformanceOptimizedPipeline


class TestModelCache(unittest.TestCase):
    """Test cases for the ModelCache class."""
    
    def setUp(self):
        """Set up test cache."""
        # Create a new cache instance for testing
        self.cache = ModelCache()
        self.cache.cache.clear()
        self.cache.access_times.clear()
    
    def test_cache_singleton(self):
        """Test that ModelCache is a singleton."""
        cache1 = ModelCache()
        cache2 = ModelCache()
        self.assertIs(cache1, cache2)
    
    def test_cache_get_with_loader(self):
        """Test cache get with loader function."""
        def mock_loader():
            return "test_model"
        
        result = self.cache.get("test_key", mock_loader)
        self.assertEqual(result, "test_model")
        self.assertIn("test_key", self.cache.cache)
    
    def test_cache_get_existing(self):
        """Test cache get for existing item."""
        # Add item to cache
        self.cache.cache["test_key"] = "cached_model"
        self.cache.access_times["test_key"] = time.time()
        
        result = self.cache.get("test_key")
        self.assertEqual(result, "cached_model")
    
    def test_cache_get_without_loader(self):
        """Test cache get without loader for non-existing item."""
        with self.assertRaises(ValueError):
            self.cache.get("non_existing_key")
    
    @patch('src.model_cache.psutil.virtual_memory')
    def test_cache_cleanup_memory_threshold(self, mock_memory):
        """Test cache cleanup when memory threshold is exceeded."""
        # Mock high memory usage
        mock_memory.return_value = Mock(percent=90)  # 90% usage
        
        # Fill cache
        for i in range(5):
            self.cache.cache[f"key_{i}"] = f"model_{i}"
            self.cache.access_times[f"key_{i}"] = time.time() - i  # Different access times
        
        initial_count = len(self.cache.cache)
        self.cache.cleanup_cache()
        
        # Should have removed some items
        self.assertLess(len(self.cache.cache), initial_count)
    
    def test_cache_cleanup_size_threshold(self):
        """Test cache cleanup when size threshold is exceeded."""
        # Fill cache beyond max size
        self.cache.max_cache_size = 3
        for i in range(5):
            self.cache.cache[f"key_{i}"] = f"model_{i}"
            self.cache.access_times[f"key_{i}"] = time.time() - i
        
        initial_count = len(self.cache.cache)
        self.cache.cleanup_cache()
        
        # Should have removed items
        self.assertLessEqual(len(self.cache.cache), self.cache.max_cache_size)
    
    @patch('src.model_cache.psutil.virtual_memory')
    def test_get_memory_usage(self, mock_memory):
        """Test memory usage calculation."""
        mock_memory.return_value = Mock(percent=75)
        
        usage = self.cache.get_memory_usage()
        self.assertEqual(usage, 0.75)
    
    def test_clear_cache(self):
        """Test cache clearing."""
        # Add items to cache
        self.cache.cache["test_key"] = "test_model"
        self.cache.access_times["test_key"] = time.time()
        
        self.cache.clear_cache()
        
        self.assertEqual(len(self.cache.cache), 0)
        self.assertEqual(len(self.cache.access_times), 0)
    
    def test_get_cache_info(self):
        """Test cache information retrieval."""
        # Add items to cache
        self.cache.cache["test_key"] = "test_model"
        
        info = self.cache.get_cache_info()
        
        self.assertIn("cached_models", info)
        self.assertIn("cache_size", info)
        self.assertIn("memory_usage", info)
        self.assertIn("max_cache_size", info)
        self.assertIn("memory_threshold", info)
        
        self.assertEqual(info["cache_size"], 1)
        self.assertIn("test_key", info["cached_models"])
    
    @patch('src.model_cache.pipeline')
    def test_preload_models(self, mock_pipeline):
        """Test model preloading."""
        mock_pipeline.return_value = Mock()
        
        try:
            self.cache.preload_models()
            # Should not raise an exception
        except Exception as e:
            self.fail(f"Preload models failed: {e}")


class TestPerformanceOptimizedPipeline(unittest.TestCase):
    """Test cases for PerformanceOptimizedPipeline."""
    
    def setUp(self):
        """Set up test pipeline."""
        self.pipeline = PerformanceOptimizedPipeline("test_cache_key")
    
    def test_pipeline_initialization(self):
        """Test pipeline initialization."""
        self.assertEqual(self.pipeline.cache_key, "test_cache_key")
        self.assertIsInstance(self.pipeline.cache, ModelCache)
    
    @patch.object(ModelCache, 'get')
    def test_get_cached_model(self, mock_get):
        """Test cached model retrieval."""
        mock_model = Mock()
        mock_get.return_value = mock_model
        
        def mock_loader():
            return "loaded_model"
        
        result = self.pipeline.get_cached_model(mock_loader)
        
        mock_get.assert_called_once()
        self.assertEqual(result, mock_model)
    
    def test_get_cached_model_with_timing(self):
        """Test cached model retrieval with timing."""
        def slow_loader():
            time.sleep(0.1)  # Simulate slow loading
            return "slow_model"
        
        start_time = time.time()
        result = self.pipeline.get_cached_model(slow_loader)
        execution_time = time.time() - start_time
        
        # Should have executed (since cache is empty)
        self.assertGreaterEqual(execution_time, 0.1)


class TestOptimizedPipelines(unittest.TestCase):
    """Test cases for optimized pipeline classes."""
    
    @patch('src.model_cache.pipeline')
    def test_optimized_sentiment_pipeline(self, mock_pipeline_func):
        """Test OptimizedSentimentPipeline."""
        from src.model_cache import OptimizedSentimentPipeline
        
        # Mock the pipeline function
        mock_classifier = Mock()
        mock_classifier.return_value = [{"label": "POSITIVE", "score": 0.9}]
        mock_pipeline_func.return_value = mock_classifier
        
        pipeline = OptimizedSentimentPipeline()
        result = pipeline("Great car!")
        
        # Should call the cached classifier
        mock_classifier.assert_called_once_with("Great car!")
        self.assertEqual(result, [{"label": "POSITIVE", "score": 0.9}])
    
    @patch('src.model_cache.pipeline')
    def test_optimized_translation_pipeline(self, mock_pipeline_func):
        """Test OptimizedTranslationPipeline."""
        from src.model_cache import OptimizedTranslationPipeline
        
        # Mock the pipeline function
        mock_translator = Mock()
        mock_translator.return_value = [{"translation_text": "Excelente coche!"}]
        mock_pipeline_func.return_value = mock_translator
        
        pipeline = OptimizedTranslationPipeline()
        result = pipeline("Great car!")
        
        # Should call the cached translator
        mock_translator.assert_called_once()
        self.assertEqual(result, "Excelente coche!")
    
    @patch('src.model_cache.pipeline')
    def test_optimized_summarization_pipeline(self, mock_pipeline_func):
        """Test OptimizedSummarizationPipeline."""
        from src.model_cache import OptimizedSummarizationPipeline
        
        # Mock the pipeline function
        mock_summarizer = Mock()
        mock_summarizer.return_value = [{"summary_text": "Great car with excellent features."}]
        mock_pipeline_func.return_value = mock_summarizer
        
        pipeline = OptimizedSummarizationPipeline()
        long_text = "This is a great car with excellent features and performance. " * 10
        result = pipeline(long_text)
        
        # Should call the cached summarizer
        mock_summarizer.assert_called_once()
        self.assertEqual(result, "Great car with excellent features.")


class TestCachePerformance(unittest.TestCase):
    """Performance tests for cache system."""
    
    def setUp(self):
        """Set up performance test cache."""
        self.cache = ModelCache()
        self.cache.cache.clear()
        self.cache.access_times.clear()
    
    def test_cache_performance_improvement(self):
        """Test that caching improves performance."""
        call_count = 0
        
        def slow_loader():
            nonlocal call_count
            call_count += 1
            time.sleep(0.1)  # Simulate slow loading
            return f"model_{call_count}"
        
        # First call (should be slow)
        start_time = time.time()
        result1 = self.cache.get("perf_test", slow_loader)
        first_call_time = time.time() - start_time
        
        # Second call (should be fast due to caching)
        start_time = time.time()
        result2 = self.cache.get("perf_test")
        second_call_time = time.time() - start_time
        
        # Verify results
        self.assertEqual(result1, result2)
        self.assertEqual(call_count, 1)  # Loader should only be called once
        self.assertGreater(first_call_time, second_call_time)
        self.assertLess(second_call_time, 0.01)  # Cache access should be very fast
    
    def test_multiple_model_caching(self):
        """Test caching multiple models simultaneously."""
        def create_loader(model_name):
            def loader():
                time.sleep(0.05)  # Simulate loading time
                return f"loaded_{model_name}"
            return loader
        
        models = ["sentiment", "translation", "summarization", "qa", "ner"]
        
        # Load all models
        start_time = time.time()
        results = {}
        for model in models:
            results[model] = self.cache.get(f"{model}_model", create_loader(model))
        loading_time = time.time() - start_time
        
        # Access all models from cache
        start_time = time.time()
        cached_results = {}
        for model in models:
            cached_results[model] = self.cache.get(f"{model}_model")
        cache_time = time.time() - start_time
        
        # Verify results
        self.assertEqual(results, cached_results)
        self.assertGreater(loading_time, cache_time)
        self.assertEqual(len(self.cache.cache), len(models))


class TestCacheEdgeCases(unittest.TestCase):
    """Test edge cases for cache system."""
    
    def setUp(self):
        """Set up edge case test cache."""
        self.cache = ModelCache()
        self.cache.cache.clear()
        self.cache.access_times.clear()
    
    def test_cache_with_exception_in_loader(self):
        """Test cache behavior when loader raises exception."""
        def failing_loader():
            raise RuntimeError("Loader failed")
        
        with self.assertRaises(RuntimeError):
            self.cache.get("failing_key", failing_loader)
        
        # Cache should not contain the failed item
        self.assertNotIn("failing_key", self.cache.cache)
    
    def test_cache_access_time_updates(self):
        """Test that access times are properly updated."""
        self.cache.cache["test_key"] = "test_model"
        
        initial_time = time.time() - 100  # Old timestamp
        self.cache.access_times["test_key"] = initial_time
        
        # Access the model
        time.sleep(0.01)  # Small delay
        result = self.cache.get("test_key")
        
        # Access time should be updated
        self.assertGreater(self.cache.access_times["test_key"], initial_time)
        self.assertEqual(result, "test_model")
    
    def test_cache_thread_safety(self):
        """Test basic thread safety of cache operations."""
        import threading
        
        results = {}
        
        def cache_operation(thread_id):
            def loader():
                time.sleep(0.01)
                return f"model_for_thread_{thread_id}"
            
            results[thread_id] = self.cache.get(f"thread_key_{thread_id}", loader)
        
        # Create multiple threads
        threads = []
        for i in range(5):
            thread = threading.Thread(target=cache_operation, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Verify results
        self.assertEqual(len(results), 5)
        for i in range(5):
            self.assertEqual(results[i], f"model_for_thread_{i}")


if __name__ == "__main__":
    unittest.main(verbosity=2)