"""
Tests for the model classes.
"""

import pytest
from unittest.mock import patch, Mock, MagicMock

from src.models import BaseModel, QuestionAnsweringModel, NamedEntityRecognitionModel


class TestBaseModel:
    """Test cases for the BaseModel class."""

    def test_base_model_init(self):
        """Test that BaseModel stores the name attribute."""
        model = BaseModel("test_model")
        assert model.name == "test_model"

    def test_base_model_call_raises(self):
        """Test that calling BaseModel raises NotImplementedError."""
        model = BaseModel("test_model")
        with pytest.raises(NotImplementedError):
            model("some input")


class TestQuestionAnsweringModel:
    """Test cases for the QuestionAnsweringModel."""

    @patch("src.models.AutoModelForQuestionAnswering.from_pretrained")
    @patch("src.models.AutoTokenizer.from_pretrained")
    def test_init_loads_model_and_tokenizer(self, mock_tokenizer_cls, mock_model_cls):
        """Test that __init__ loads tokenizer and model from the checkpoint."""
        mock_tokenizer_cls.return_value = MagicMock()
        mock_model_cls.return_value = MagicMock()

        model = QuestionAnsweringModel(model_checkpoint="test-checkpoint")

        mock_tokenizer_cls.assert_called_once_with("test-checkpoint")
        mock_model_cls.assert_called_once_with("test-checkpoint")
        assert model.name == "Question Answering"
        assert model.model_ckp == "test-checkpoint"

    @patch("src.models.torch")
    @patch("src.models.AutoModelForQuestionAnswering.from_pretrained")
    @patch("src.models.AutoTokenizer.from_pretrained")
    def test_call_returns_decoded_answer(self, mock_tokenizer_cls, mock_model_cls, mock_torch):
        """Test that __call__ tokenizes input, runs model, and decodes the answer."""
        # Set up tokenizer mock
        mock_tokenizer = MagicMock()
        mock_tokenizer_cls.return_value = mock_tokenizer

        mock_input_ids = MagicMock()
        mock_input_ids_row = MagicMock()
        mock_input_ids_row.__getitem__ = Mock(return_value=MagicMock())
        mock_input_ids.__getitem__ = Mock(return_value=mock_input_ids_row)

        mock_inputs = MagicMock()
        mock_inputs.__getitem__ = Mock(return_value=mock_input_ids)
        mock_tokenizer.return_value = mock_inputs

        mock_tokenizer.decode.return_value = "350 miles"

        # Set up model mock
        mock_model = MagicMock()
        mock_model_cls.return_value = mock_model

        mock_outputs = MagicMock()
        mock_outputs.start_logits = MagicMock()
        mock_outputs.end_logits = MagicMock()
        mock_model.return_value = mock_outputs

        # torch.argmax must return real ints (source does argmax + 1)
        mock_torch.argmax.side_effect = [2, 4]
        mock_torch.no_grad.return_value.__enter__ = Mock(return_value=None)
        mock_torch.no_grad.return_value.__exit__ = Mock(return_value=False)

        # Create model and call
        qa_model = QuestionAnsweringModel(model_checkpoint="test-ckpt")
        answer = qa_model("This car has 350 miles range.", "What is the range?")

        assert answer == "350 miles"
        mock_tokenizer.decode.assert_called_once()

    @patch("src.models.AutoModelForQuestionAnswering.from_pretrained")
    @patch("src.models.AutoTokenizer.from_pretrained")
    def test_init_raises_on_model_load_failure(self, mock_tok, mock_model):
        """Test that __init__ propagates exceptions from model loading."""
        mock_model.side_effect = OSError("Model not found")

        with pytest.raises(OSError, match="Model not found"):
            QuestionAnsweringModel(model_checkpoint="bad-model")


class TestNamedEntityRecognitionModel:
    """Test cases for the NamedEntityRecognitionModel."""

    @patch("src.models.pipeline")
    @patch("src.models.ENABLE_MODEL_CACHE", False, create=True)
    def test_init_without_cache(self, mock_pipeline_fn):
        """Test that __init__ loads NER pipeline directly when cache is disabled."""
        mock_ner = MagicMock()
        mock_pipeline_fn.return_value = mock_ner

        # Patch the inline imports inside __init__
        with patch.dict(
            "sys.modules",
            {
                "src.model_cache": MagicMock(),
                "src.config": MagicMock(ENABLE_MODEL_CACHE=False),
            },
        ):
            # Re-patch pipeline at module level since it's used directly
            with patch("src.models.pipeline", return_value=mock_ner):
                model = NamedEntityRecognitionModel()

        assert model.name == "Named Entity Recognition"

    @patch("src.models.pipeline")
    def test_call_groups_entities_and_filters(self, mock_pipeline_fn):
        """Test that __call__ groups BIO-tagged entities and filters low-confidence ones."""
        mock_ner = MagicMock()
        mock_pipeline_fn.return_value = mock_ner

        # Bypass cache in __init__
        with patch("src.models.ENABLE_MODEL_CACHE", False, create=True):
            with patch.dict(
                "sys.modules",
                {
                    "src.model_cache": MagicMock(),
                    "src.config": MagicMock(ENABLE_MODEL_CACHE=False),
                },
            ):
                with patch("src.models.pipeline", return_value=mock_ner):
                    model = NamedEntityRecognitionModel()

        # Set up NER mock return
        model.ner = MagicMock()
        model.ner.return_value = [
            {"word": "Tesla", "entity": "B-ORG", "score": 0.95, "start": 0, "end": 5},
            {"word": "Model", "entity": "I-ORG", "score": 0.90, "start": 6, "end": 11},
            {"word": "3", "entity": "I-ORG", "score": 0.88, "start": 12, "end": 13},
            {"word": "weak", "entity": "B-MISC", "score": 0.3, "start": 20, "end": 24},
        ]

        result = model("Tesla Model 3 is weak")

        # Should group Tesla+Model+3 into one entity, filter out "weak" (score 0.3)
        assert len(result) == 1
        assert result[0]["entity"] == "ORG"
        assert "Tesla" in result[0]["word"]
        assert result[0]["score"] > 0.7

    @patch("src.models.pipeline")
    def test_call_returns_empty_on_error(self, mock_pipeline_fn):
        """Test that __call__ returns empty list on exception instead of raising."""
        mock_ner = MagicMock()
        mock_pipeline_fn.return_value = mock_ner

        with patch("src.models.ENABLE_MODEL_CACHE", False, create=True):
            with patch.dict(
                "sys.modules",
                {
                    "src.model_cache": MagicMock(),
                    "src.config": MagicMock(ENABLE_MODEL_CACHE=False),
                },
            ):
                with patch("src.models.pipeline", return_value=mock_ner):
                    model = NamedEntityRecognitionModel()

        model.ner = MagicMock(side_effect=RuntimeError("NER failed"))

        result = model("some text")
        assert result == []

    @patch("src.models.pipeline")
    def test_call_empty_input(self, mock_pipeline_fn):
        """Test that __call__ handles empty text input gracefully."""
        mock_ner = MagicMock()
        mock_pipeline_fn.return_value = mock_ner

        with patch("src.models.ENABLE_MODEL_CACHE", False, create=True):
            with patch.dict(
                "sys.modules",
                {
                    "src.model_cache": MagicMock(),
                    "src.config": MagicMock(ENABLE_MODEL_CACHE=False),
                },
            ):
                with patch("src.models.pipeline", return_value=mock_ner):
                    model = NamedEntityRecognitionModel()

        model.ner = MagicMock(return_value=[])

        result = model("")
        assert result == []
