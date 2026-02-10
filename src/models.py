"""
Model classes for car review analysis tasks.

This module provides model classes for various NLP tasks:
- Question Answering: Extract answers from reviews based on questions
- Zero-Shot Classification: Classify reviews without specific training
- Named Entity Recognition: Extract entities like car brands and models
"""

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForQuestionAnswering,
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    pipeline,
)
from typing import List, Dict, Any, Union, Optional

from src.logger import get_logger
from src.config import QA_MODEL

# Set up logger
logger = get_logger(__name__)


class BaseModel:
    """Base class for all NLP models."""

    def __init__(self, name: str):
        """
        Initialize the base model.

        Args:
            name: Name of the model.
        """
        self.name = name
        logger.info(f"Initializing {name} model")

    def __call__(self, *args, **kwargs):
        """
        Process inputs through the model.

        This method should be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement __call__")


class QuestionAnsweringModel(BaseModel):
    """Model for answering questions about car reviews."""

    def __init__(self, model_checkpoint: str = QA_MODEL):
        """
        Initialize the question answering model.

        Args:
            model_checkpoint: Pretrained model checkpoint.
        """
        super().__init__("Question Answering")
        try:
            self.model_ckp = model_checkpoint
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_ckp)
            self.model = AutoModelForQuestionAnswering.from_pretrained(self.model_ckp)
            logger.info(f"Loaded question answering model: {model_checkpoint}")
        except Exception as e:
            logger.error(f"Failed to load question answering model: {e}")
            raise

    def __call__(self, context: str, question: str) -> str:
        """
        Answer a question based on the context.

        Args:
            context: The context text (car review).
            question: The question to answer.

        Returns:
            The answer extracted from the context.
        """
        logger.info(f"Answering question: '{question}'")
        try:
            inputs = self.tokenizer(
                question, context, return_tensors="pt", truncation=True, max_length=512
            )
            with torch.no_grad():
                outputs = self.model(**inputs)

            start_idx = torch.argmax(outputs.start_logits)
            end_idx = torch.argmax(outputs.end_logits) + 1
            answer_span = inputs["input_ids"][0][start_idx:end_idx]
            answer = self.tokenizer.decode(answer_span)

            logger.info(f"Generated answer: '{answer}'")
            return answer
        except Exception as e:
            logger.error(f"Error during question answering: {e}")
            raise


# NOTE: Removed unused models (ZeroShotClassificationModel, NamedEntityRecognitionModel, SentimentIntensityModel)
# These models were not integrated into the main pipeline and added unnecessary complexity.
# If needed in the future, they can be re-implemented with proper integration.


class NamedEntityRecognitionModel(BaseModel):
    """Enhanced NER model with caching support."""

    def __init__(self):
        """Initialize the named entity recognition model."""
        super().__init__("Named Entity Recognition")
        try:
            from src.model_cache import model_cache
            from src.config import ENABLE_MODEL_CACHE

            if ENABLE_MODEL_CACHE:
                self.ner = model_cache.get("ner_pipeline", lambda: pipeline("ner"))
            else:
                self.ner = pipeline("ner")
            logger.info("Loaded named entity recognition model")
        except Exception as e:
            logger.error(f"Failed to load named entity recognition model: {e}")
            raise

    def __call__(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract named entities from text with enhanced grouping.

        Args:
            text: The text to analyze.

        Returns:
            List of extracted entities.
        """
        logger.info(f"Extracting named entities from text (length: {len(text)})")
        try:
            entities = self.ner(text)

            # Enhanced entity grouping
            grouped_entities = []
            current_entity = None

            for entity in entities:
                if current_entity is None or entity["entity"].startswith("B-"):
                    if current_entity is not None:
                        grouped_entities.append(current_entity)
                    current_entity = {
                        "word": entity["word"],
                        "entity": entity["entity"].split("-")[-1],  # Get entity type
                        "score": entity["score"],
                        "start": entity["start"],
                        "end": entity["end"],
                        "_token_count": 1,
                    }
                else:
                    # Continue building the entity
                    word_part = entity["word"].replace("##", "")
                    current_entity["word"] += word_part
                    current_entity["end"] = entity["end"]
                    n = current_entity["_token_count"]
                    current_entity["score"] = (
                        current_entity["score"] * n + entity["score"]
                    ) / (n + 1)
                    current_entity["_token_count"] = n + 1

            if current_entity is not None:
                current_entity.pop("_token_count", None)
                grouped_entities.append(current_entity)

            # Clean up internal tracking keys
            for entity in grouped_entities:
                entity.pop("_token_count", None)

            # Filter out low-confidence entities
            filtered_entities = [e for e in grouped_entities if e["score"] > 0.7]

            logger.info(
                f"Extracted {len(filtered_entities)} high-confidence entities from {len(grouped_entities)} total"
            )
            return filtered_entities
        except Exception as e:
            logger.error(f"Error during named entity recognition: {e}")
            return []  # Return empty list instead of raising
