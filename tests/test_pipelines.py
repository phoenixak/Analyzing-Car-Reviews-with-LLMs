"""
Enhanced comprehensive tests for the pipeline classes.
"""

import unittest
import sys
import os
from pathlib import Path
from unittest.mock import Mock, patch

# Add the parent directory to the path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.pipelines import (
    SentimentAnalysisPipeline,
    TranslationPipeline,
    SummarizationPipeline,
    TopicModelingPipeline,
    AspectSentimentPipeline,
)
from src.error_handler import ModelLoadError, DataProcessingError


class TestEnhancedPipelines(unittest.TestCase):
    """Enhanced test cases for the pipeline classes."""

    def setUp(self):
        """Set up test data."""
        self.test_review = "I really enjoyed driving this car. The handling was excellent and the fuel economy was great. However, the price was a bit high."
        self.test_reviews = [
            "Great car with excellent performance and comfort.",
            "Poor fuel economy and expensive maintenance costs.",
            "Amazing design but lacking in safety features.",
            "Best car I've ever owned, highly recommended.",
            "Terrible experience, would not buy again."
        ]
        
    @patch('src.pipelines.model_cache')
    @patch('src.pipelines.ENABLE_MODEL_CACHE', True)
    def test_sentiment_analysis_pipeline_with_cache(self, mock_cache):
        """Test sentiment analysis pipeline with caching enabled."""
        # Mock the cached model
        mock_classifier = Mock()
        mock_classifier.return_value = [{"label": "POSITIVE", "score": 0.95}]
        mock_cache.get.return_value = mock_classifier
        
        try:
            pipeline = SentimentAnalysisPipeline()
            result = pipeline(self.test_review)

            # Verify cache was used
            mock_cache.get.assert_called()
            
            # Check result format
            self.assertIsInstance(result, list)
            self.assertTrue(len(result) > 0)
            self.assertIn("label", result[0])
            self.assertIn("score", result[0])
            self.assertIn(result[0]["label"], ["POSITIVE", "NEGATIVE"])
            self.assertTrue(0 <= result[0]["score"] <= 1)
            
        except Exception as e:
            self.fail(f"Sentiment analysis pipeline with cache raised an exception: {e}")

    def test_sentiment_analysis_empty_input(self):
        """Test sentiment analysis with empty input."""
        pipeline = SentimentAnalysisPipeline()
        result = pipeline("")
        
        # Should handle empty input gracefully
        self.assertIsInstance(result, list)

    def test_sentiment_analysis_multiple_reviews(self):
        """Test sentiment analysis with multiple reviews."""
        pipeline = SentimentAnalysisPipeline()
        result = pipeline(self.test_reviews)
        
        # Should return results for all reviews
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), len(self.test_reviews))

    @patch('src.pipelines.safe_model_load')
    def test_translation_pipeline_with_error_handling(self, mock_load):
        """Test translation pipeline with error handling."""
        mock_load.side_effect = ModelLoadError("Model failed to load")
        
        try:
            pipeline = TranslationPipeline()
            # Should handle model load error gracefully
            # (depends on error handler implementation)
        except ModelLoadError:
            pass  # Expected behavior

    def test_topic_modeling_insufficient_data(self):
        """Test topic modeling with insufficient data."""
        pipeline = TopicModelingPipeline(min_topic_size=5)
        small_dataset = ["Short review"]
        
        result = pipeline(small_dataset)
        
        # Should handle insufficient data gracefully
        self.assertIsInstance(result, dict)
        self.assertIn("topic_info", result)

    def test_topic_modeling_normal_operation(self):
        """Test topic modeling with sufficient data."""
        pipeline = TopicModelingPipeline(num_topics=3, min_topic_size=2)
        
        result = pipeline(self.test_reviews)
        
        # Check result structure
        self.assertIsInstance(result, dict)
        self.assertIn("topics", result)
        self.assertIn("topic_words", result)
        self.assertIn("document_topics", result)
        self.assertIn("num_topics", result)

    def test_aspect_sentiment_pipeline(self):
        """Test aspect-based sentiment analysis pipeline."""
        pipeline = AspectSentimentPipeline()
        
        result = pipeline(self.test_review)
        
        # Check result structure
        self.assertIsInstance(result, dict)
        
        # Should have results for predefined aspects
        expected_aspects = ["price", "performance", "comfort", "reliability", "design", "safety", "fuel economy"]
        for aspect in expected_aspects:
            if aspect in result:
                self.assertIn("positive", result[aspect])
                self.assertIn("negative", result[aspect])
                self.assertIn("neutral", result[aspect])

    def test_pipeline_error_recovery(self):
        """Test pipeline error recovery mechanisms."""
        # Test with various error conditions
        pipeline = SentimentAnalysisPipeline()
        
        # Test with None input
        result = pipeline(None)
        self.assertIsInstance(result, list)
        
        # Test with very long text
        very_long_text = "test " * 10000
        result = pipeline(very_long_text)
        self.assertIsInstance(result, list)

    def test_summarization_edge_cases(self):
        """Test summarization with edge cases."""
        pipeline = SummarizationPipeline()
        
        # Test with very short text
        short_text = "Good car."
        result = pipeline(short_text)
        self.assertIsInstance(result, str)
        
        # Test with very long text
        long_text = self.test_review * 10
        result = pipeline(long_text)
        self.assertIsInstance(result, str)
        self.assertTrue(len(result) < len(long_text))

    @patch('src.pipelines.BERTopic')
    def test_topic_modeling_import_error(self, mock_bertopic):
        """Test topic modeling when BERTopic is not available."""
        mock_bertopic.side_effect = ImportError("BERTopic not installed")
        
        with self.assertRaises(ImportError):
            TopicModelingPipeline()

    def test_translation_various_lengths(self):
        """Test translation with various text lengths."""
        pipeline = TranslationPipeline()
        
        # Test different text lengths
        test_cases = [
            "Car",
            "Good car with nice features.",
            self.test_review,
            " ".join(self.test_reviews)
        ]
        
        for text in test_cases:
            result = pipeline(text)
            self.assertIsInstance(result, str)
            self.assertTrue(len(result) > 0)

    def test_pipeline_performance(self):
        """Test pipeline performance with timing."""
        import time
        
        pipeline = SentimentAnalysisPipeline()
        
        start_time = time.time()
        result = pipeline(self.test_reviews)
        execution_time = time.time() - start_time
        
        # Should complete reasonably quickly (adjust threshold as needed)
        self.assertLess(execution_time, 30.0)  # 30 seconds max
        self.assertIsInstance(result, list)

    def tearDown(self):
        """Clean up after tests."""
        # Clear any cached models if needed
        try:
            from src.model_cache import model_cache
            if hasattr(model_cache, 'clear_cache'):
                model_cache.clear_cache()
        except ImportError:
            pass


class TestPipelineIntegration(unittest.TestCase):
    """Integration tests for pipeline combinations."""
    
    def setUp(self):
        """Set up integration test data."""
        self.sample_reviews = [
            "Excellent car with great performance and comfort features.",
            "Poor fuel economy and high maintenance costs disappointed me.",
            "Beautiful design but lacks some important safety features."
        ]
    
    def test_full_pipeline_integration(self):
        """Test running multiple pipelines on the same data."""
        try:
            # Initialize all pipelines
            sentiment_pipeline = SentimentAnalysisPipeline()
            translation_pipeline = TranslationPipeline()
            summarization_pipeline = SummarizationPipeline()
            
            results = {}
            
            # Run sentiment analysis
            results['sentiment'] = sentiment_pipeline(self.sample_reviews)
            
            # Run translation on first review
            results['translation'] = translation_pipeline(self.sample_reviews[0])
            
            # Run summarization on combined reviews
            combined_text = " ".join(self.sample_reviews)
            results['summary'] = summarization_pipeline(combined_text)
            
            # Verify all results
            self.assertIn('sentiment', results)
            self.assertIn('translation', results)
            self.assertIn('summary', results)
            
            self.assertIsInstance(results['sentiment'], list)
            self.assertIsInstance(results['translation'], str)
            self.assertIsInstance(results['summary'], str)
            
        except Exception as e:
            self.fail(f"Full pipeline integration test failed: {e}")
    
    def test_pipeline_data_flow(self):
        """Test data flow between different pipeline stages."""
        # Test processing pipeline where output of one stage feeds into another
        sentiment_pipeline = SentimentAnalysisPipeline()
        
        # Get sentiment results
        sentiment_results = sentiment_pipeline(self.sample_reviews)
        
        # Use sentiment results to filter reviews
        positive_reviews = [
            self.sample_reviews[i] for i, result in enumerate(sentiment_results)
            if result.get('label') == 'POSITIVE'
        ]
        
        if positive_reviews:
            # Process positive reviews further
            summary_pipeline = SummarizationPipeline()
            summary = summary_pipeline(" ".join(positive_reviews))
            
            self.assertIsInstance(summary, str)
            self.assertTrue(len(summary) > 0)


if __name__ == "__main__":
    # Run with higher verbosity for better output
    unittest.main(verbosity=2)
