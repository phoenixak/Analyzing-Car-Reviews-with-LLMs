"""
Tests for the pipeline classes.
"""

import sys

import pytest
from unittest.mock import Mock, MagicMock, patch

from src.pipelines import (
    BasePipeline,
    SentimentAnalysisPipeline,
    TranslationPipeline,
    SummarizationPipeline,
    TopicModelingPipeline,
    AspectSentimentPipeline,
)


class TestBasePipeline:
    """Test cases for the BasePipeline class."""

    def test_init_stores_name(self):
        """Test that BasePipeline stores the pipeline name."""
        pipe = BasePipeline("test_pipeline")
        assert pipe.name == "test_pipeline"

    def test_call_raises_not_implemented(self):
        """Test that calling BasePipeline raises NotImplementedError."""
        pipe = BasePipeline("test_pipeline")
        with pytest.raises(NotImplementedError):
            pipe("input")


class TestSentimentAnalysisPipeline:
    """Test cases for the SentimentAnalysisPipeline."""

    @patch("src.pipelines.ENABLE_MODEL_CACHE", True)
    @patch("src.pipelines.model_cache")
    def test_sentiment_with_cache_single_review(self, mock_cache):
        """Test sentiment analysis on a single review with caching enabled."""
        mock_classifier = MagicMock()
        # Real HF pipelines return a single dict (not list) for string input.
        # The source code then wraps it: if isinstance(reviews, str): results = [results]
        mock_classifier.return_value = {"label": "POSITIVE", "score": 0.95}
        mock_cache.get.return_value = mock_classifier

        pipe = SentimentAnalysisPipeline()
        result = pipe("Great car!")

        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0]["label"] == "POSITIVE"

    @patch("src.pipelines.ENABLE_MODEL_CACHE", True)
    @patch("src.pipelines.model_cache")
    def test_sentiment_with_cache_multiple_reviews(self, mock_cache):
        """Test sentiment analysis on a list of reviews with caching enabled."""
        mock_classifier = MagicMock()
        mock_classifier.return_value = [
            {"label": "POSITIVE", "score": 0.95},
            {"label": "NEGATIVE", "score": 0.88},
        ]
        mock_cache.get.return_value = mock_classifier

        pipe = SentimentAnalysisPipeline()
        result = pipe(["Great car!", "Terrible car."])

        assert isinstance(result, list)
        assert len(result) == 2

    @patch("src.pipelines.ENABLE_MODEL_CACHE", True)
    @patch("src.pipelines.model_cache")
    def test_sentiment_empty_input(self, mock_cache):
        """Test sentiment analysis returns empty list for empty input."""
        mock_cache.get.return_value = MagicMock()

        pipe = SentimentAnalysisPipeline()
        result = pipe("")

        assert result == []

    @patch("src.pipelines.safe_model_load")
    @patch("src.pipelines.ENABLE_MODEL_CACHE", False)
    @patch("src.pipelines.model_cache")
    def test_sentiment_without_cache(self, mock_cache, mock_safe_load):
        """Test sentiment analysis with caching disabled."""
        mock_classifier = MagicMock()
        mock_classifier.return_value = [{"label": "POSITIVE", "score": 0.9}]
        mock_safe_load.return_value = mock_classifier

        pipe = SentimentAnalysisPipeline()
        result = pipe("Nice ride.")

        assert isinstance(result, list)
        assert len(result) == 1


class TestTranslationPipeline:
    """Test cases for the TranslationPipeline."""

    @patch("src.pipelines.pipeline")
    def test_translation_init_and_call(self, mock_pipeline_fn):
        """Test translation pipeline loads model and translates text."""
        mock_translator = MagicMock()
        mock_translator.return_value = [{"translation_text": "Excelente coche!"}]
        mock_pipeline_fn.return_value = mock_translator

        pipe = TranslationPipeline()
        result = pipe("Great car!")

        mock_pipeline_fn.assert_called_once_with("translation", model="Helsinki-NLP/opus-mt-en-es")
        assert result == "Excelente coche!"

    @patch("src.pipelines.pipeline")
    def test_translation_error_propagation(self, mock_pipeline_fn):
        """Test that translation errors propagate as exceptions."""
        mock_translator = MagicMock()
        mock_translator.side_effect = RuntimeError("Translation failed")
        mock_pipeline_fn.return_value = mock_translator

        pipe = TranslationPipeline()
        with pytest.raises(RuntimeError, match="Translation failed"):
            pipe("Some text")

    @patch("src.pipelines.pipeline")
    def test_translation_init_failure(self, mock_pipeline_fn):
        """Test that a model load failure in __init__ propagates."""
        mock_pipeline_fn.side_effect = OSError("Cannot download model")

        with pytest.raises(OSError):
            TranslationPipeline()


class TestSummarizationPipeline:
    """Test cases for the SummarizationPipeline."""

    @patch("src.pipelines.pipeline")
    def test_summarization_init_and_call(self, mock_pipeline_fn):
        """Test summarization pipeline loads model and summarizes text."""
        mock_summarizer = MagicMock()
        mock_summarizer.return_value = [{"summary_text": "Great car with excellent features."}]
        mock_pipeline_fn.return_value = mock_summarizer

        pipe = SummarizationPipeline()
        long_text = "This is a great car with excellent features and performance. " * 10
        result = pipe(long_text)

        assert result == "Great car with excellent features."

    @patch("src.pipelines.pipeline")
    def test_summarization_error_propagation(self, mock_pipeline_fn):
        """Test that summarization errors propagate as exceptions."""
        mock_summarizer = MagicMock()
        mock_summarizer.side_effect = RuntimeError("Summarization failed")
        mock_pipeline_fn.return_value = mock_summarizer

        pipe = SummarizationPipeline()
        with pytest.raises(RuntimeError, match="Summarization failed"):
            pipe("Some text")


class TestTopicModelingPipeline:
    """Test cases for the TopicModelingPipeline."""

    def test_insufficient_data(self):
        """Test topic modeling returns fallback dict when data is too small."""
        mock_bt = MagicMock()
        mock_st = MagicMock()

        with patch.dict("sys.modules", {"bertopic": mock_bt, "sentence_transformers": mock_st}):
            pipe = TopicModelingPipeline(num_topics=2, min_topic_size=5)
            result = pipe(["Short review"])

        assert isinstance(result, dict)
        assert result["topic_info"] == "Insufficient data for topic modeling"
        assert result["topics"] == []

    def test_normal_operation(self):
        """Test topic modeling with mocked BERTopic produces expected result structure."""
        mock_bt = MagicMock()
        mock_st = MagicMock()

        with patch.dict("sys.modules", {"bertopic": mock_bt, "sentence_transformers": mock_st}):
            pipe = TopicModelingPipeline(num_topics=2, min_topic_size=2)

            # Configure the mock topic model
            pipe.topic_model.fit_transform.return_value = (
                [0, 1, 0, 1],
                [[0.9, 0.1], [0.1, 0.9], [0.8, 0.2], [0.2, 0.8]],
            )

            mock_info = MagicMock()
            mock_info.__len__ = Mock(return_value=2)
            mock_info.to_dict.return_value = [{"Topic": 0}, {"Topic": 1}]
            pipe.topic_model.get_topic_info.return_value = mock_info

            pipe.topic_model.get_topic.return_value = [
                ("great", 0.9),
                ("car", 0.8),
                ("performance", 0.7),
                ("comfort", 0.6),
                ("price", 0.5),
            ]

            texts = ["review one", "review two", "review three", "review four"]
            result = pipe(texts)

        assert isinstance(result, dict)
        assert "topics" in result
        assert "topic_words" in result
        assert "document_topics" in result
        assert len(result["document_topics"]) == 4

    def test_init_import_error(self):
        """Test that missing BERTopic raises ImportError."""
        # Remove bertopic from sys.modules if present, then import
        mock_bt = MagicMock()
        mock_bt.BERTopic.side_effect = ImportError("BERTopic not installed")

        # The real check: the inline import itself should fail
        with patch.dict(
            "sys.modules",
            {"bertopic": None, "sentence_transformers": MagicMock()},
        ):
            with pytest.raises(ImportError):
                TopicModelingPipeline()

    def test_call_error_returns_fallback(self):
        """Test that __call__ returns a fallback dict on exception."""
        mock_bt = MagicMock()
        mock_st = MagicMock()

        with patch.dict("sys.modules", {"bertopic": mock_bt, "sentence_transformers": mock_st}):
            pipe = TopicModelingPipeline(num_topics=2, min_topic_size=2)

            pipe.topic_model.fit_transform.side_effect = RuntimeError("fit failed")

            result = pipe(["review 1", "review 2", "review 3"])

        assert isinstance(result, dict)
        assert "Error" in result["topic_info"]
        assert result["num_topics"] == 0


class TestAspectSentimentPipeline:
    """Test cases for the AspectSentimentPipeline."""

    @patch("src.pipelines.AutoModelForSequenceClassification.from_pretrained")
    @patch("src.pipelines.AutoTokenizer.from_pretrained")
    def test_init_loads_model(self, mock_tokenizer_cls, mock_model_cls):
        """Test that __init__ loads tokenizer and model."""
        mock_tokenizer_cls.return_value = MagicMock()
        mock_model_cls.return_value = MagicMock()

        pipe = AspectSentimentPipeline()

        mock_tokenizer_cls.assert_called_once()
        mock_model_cls.assert_called_once()
        assert pipe.name == "Aspect-Based Sentiment Analysis"
        assert len(pipe.aspects) == 7

    @patch("src.pipelines.torch")
    @patch("src.pipelines.AutoModelForSequenceClassification.from_pretrained")
    @patch("src.pipelines.AutoTokenizer.from_pretrained")
    def test_call_returns_aspect_scores(self, mock_tokenizer_cls, mock_model_cls, mock_torch):
        """Test that __call__ returns sentiment scores for each aspect."""
        # Set up tokenizer
        mock_tokenizer = MagicMock()
        mock_tokenizer_cls.return_value = mock_tokenizer

        # Set up model
        mock_model = MagicMock()
        mock_model_cls.return_value = mock_model

        mock_outputs = MagicMock()
        mock_outputs.logits = MagicMock()
        mock_model.return_value = mock_outputs

        # Set up torch mocks
        mock_torch.no_grad.return_value.__enter__ = Mock(return_value=None)
        mock_torch.no_grad.return_value.__exit__ = Mock(return_value=False)

        # Build softmax return: probs[0], probs[1], probs[2] -> .item()
        def make_prob(val):
            m = Mock()
            m.item.return_value = val
            return m

        mock_probs = MagicMock()
        mock_probs.__getitem__ = Mock(side_effect=lambda i: make_prob([0.1, 0.2, 0.7][i]))

        mock_softmax_result = MagicMock()
        mock_softmax_result.__getitem__ = Mock(return_value=mock_probs)

        mock_torch.nn.functional.softmax.return_value = mock_softmax_result

        pipe = AspectSentimentPipeline()
        result = pipe("Great car with excellent performance.")

        assert isinstance(result, dict)
        expected_aspects = [
            "price",
            "performance",
            "comfort",
            "reliability",
            "design",
            "safety",
            "fuel economy",
        ]
        for aspect in expected_aspects:
            assert aspect in result
            assert "positive" in result[aspect]
            assert "negative" in result[aspect]
            assert "neutral" in result[aspect]
            assert result[aspect]["positive"] == 0.7
            assert result[aspect]["negative"] == 0.1
            assert result[aspect]["neutral"] == 0.2

    @patch("src.pipelines.AutoModelForSequenceClassification.from_pretrained")
    @patch("src.pipelines.AutoTokenizer.from_pretrained")
    def test_init_failure(self, mock_tok, mock_model):
        """Test that model load failure in __init__ propagates."""
        mock_model.side_effect = OSError("Model not found")

        with pytest.raises(OSError):
            AspectSentimentPipeline()
