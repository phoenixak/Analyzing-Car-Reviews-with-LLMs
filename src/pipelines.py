"""
Pipeline classes for car review analysis tasks.

This module provides pipeline classes for various NLP tasks:
- Sentiment Analysis: Classify reviews as positive or negative
- Translation: Translate reviews from English to Spanish
- Summarization: Summarize long reviews
- Topic Modeling: Extract topics from reviews
- Aspect-Based Sentiment Analysis: Analyze sentiment for specific aspects
"""

from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
import torch
from typing import List, Dict, Any, Union, Optional
import numpy as np

from src.logger import get_logger
from src.config import (
    SENTIMENT_MODEL,
    TRANSLATION_MODEL,
    SUMMARIZATION_MODEL,
    TOPIC_MODEL,
    ASPECT_SENTIMENT_MODEL,
    ENABLE_MODEL_CACHE,
    MIN_TOPIC_SIZE,
    NUM_TOPICS
)
from src.error_handler import (
    robust_operation,
    safe_model_load,
    ModelLoadError,
    DataProcessingError,
    error_handler
)
from src.model_cache import model_cache

# Set up logger
logger = get_logger(__name__)


class BasePipeline:
    """Base class for all NLP pipelines."""

    def __init__(self, name: str):
        """
        Initialize the base pipeline.

        Args:
            name: Name of the pipeline.
        """
        self.name = name
        logger.info(f"Initializing {name} pipeline")

    def __call__(self, *args, **kwargs):
        """
        Process inputs through the pipeline.

        This method should be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement __call__")


class SentimentAnalysisPipeline(BasePipeline):
    """Enhanced pipeline for sentiment analysis of car reviews with caching and error handling."""

    def __init__(self, model_name: str = SENTIMENT_MODEL):
        """
        Initialize the sentiment analysis pipeline.

        Args:
            model_name: Name of the pre-trained model to use.
        """
        super().__init__("Sentiment Analysis")
        self.model_name = model_name
        self.cache_key = f"sentiment_{model_name}"
        
        if ENABLE_MODEL_CACHE:
            # Pre-cache the model for better performance
            try:
                self._get_classifier()
                logger.info(f"Pre-cached sentiment analysis model: {model_name}")
            except Exception as e:
                logger.warning(f"Failed to pre-cache model, will load on demand: {e}")

    def _get_classifier(self):
        """Get the sentiment classifier with caching."""
        if ENABLE_MODEL_CACHE:
            return model_cache.get(
                self.cache_key,
                lambda: safe_model_load(
                    pipeline, 
                    self.model_name,
                    task="sentiment-analysis", 
                    model=self.model_name
                )
            )
        else:
            return safe_model_load(
                pipeline,
                self.model_name,
                task="sentiment-analysis",
                model=self.model_name
            )

    @robust_operation(fallback_value=[], context="sentiment_analysis")
    def __call__(self, reviews: Union[str, List[str]]) -> List[Dict[str, Any]]:
        """
        Analyze sentiment of car reviews with robust error handling.

        Args:
            reviews: Single review or list of reviews.

        Returns:
            List of dictionaries containing sentiment analysis results.
        """
        logger.info(
            f"Analyzing sentiment of {'multiple reviews' if isinstance(reviews, list) else 'a review'}"
        )
        
        if not reviews:
            logger.warning("No reviews provided for sentiment analysis")
            return []
        
        try:
            classifier = self._get_classifier()
            
            if not classifier:
                raise ModelLoadError(f"Failed to load sentiment classifier: {self.model_name}")
            
            # Process reviews
            results = classifier(reviews)
            if isinstance(reviews, str):
                results = [results]
            
            # Validate results
            if not isinstance(results, list):
                raise DataProcessingError("Sentiment analysis returned invalid format")
            
            logger.info(f"Successfully analyzed sentiment for {len(results)} review(s)")
            return results
            
        except Exception as e:
            error_handler.handle_error(e, "sentiment_analysis")
            # Return empty results as fallback
            return [{"label": "UNKNOWN", "score": 0.0} for _ in (reviews if isinstance(reviews, list) else [reviews])]


class TranslationPipeline(BasePipeline):
    """Pipeline for translating car reviews from English to Spanish."""

    def __init__(self, model_name: str = TRANSLATION_MODEL):
        """
        Initialize the translation pipeline.

        Args:
            model_name: Name of the pre-trained model to use.
        """
        super().__init__("Translation")
        try:
            self.translator = pipeline("translation", model=model_name)
            logger.info(f"Loaded translation model: {model_name}")
        except Exception as e:
            logger.error(f"Failed to load translation model: {e}")
            raise

    def __call__(self, text: str, max_length: int = 512) -> str:
        """
        Translate car reviews from English to Spanish.

        Args:
            text: Text to translate.
            max_length: Maximum length of the translated text.

        Returns:
            Translated text.
        """
        logger.info(f"Translating text (length: {len(text)})")
        try:
            result = self.translator(text, max_length=max_length)
            return result[0]["translation_text"]
        except Exception as e:
            logger.error(f"Error during translation: {e}")
            raise


class SummarizationPipeline(BasePipeline):
    """Pipeline for summarizing car reviews."""

    def __init__(self, model_name: str = SUMMARIZATION_MODEL):
        """
        Initialize the summarization pipeline.

        Args:
            model_name: Name of the pre-trained model to use.
        """
        super().__init__("Summarization")
        try:
            self.summarizer = pipeline("summarization", model=model_name)
            logger.info(f"Loaded summarization model: {model_name}")
        except Exception as e:
            logger.error(f"Failed to load summarization model: {e}")
            raise

    def __call__(self, text: str, max_length: int = 100, min_length: int = 30) -> str:
        """
        Summarize car reviews.

        Args:
            text: Text to summarize.
            max_length: Maximum length of the summary.
            min_length: Minimum length of the summary.

        Returns:
            Summarized text.
        """
        logger.info(f"Summarizing text (length: {len(text)})")
        try:
            result = self.summarizer(text, max_length=max_length, min_length=min_length)
            return result[0]["summary_text"]
        except Exception as e:
            logger.error(f"Error during summarization: {e}")
            raise


class TopicModelingPipeline(BasePipeline):
    """Pipeline for topic modeling of car reviews using BERTopic."""

    def __init__(self, num_topics: int = NUM_TOPICS, min_topic_size: int = MIN_TOPIC_SIZE):
        """
        Initialize the topic modeling pipeline.

        Args:
            num_topics: Number of topics to extract (used as guidance).
            min_topic_size: Minimum size of topics.
        """
        super().__init__("Topic Modeling")
        self.num_topics = num_topics
        self.min_topic_size = min_topic_size
        self.topic_model = None
        
        try:
            from bertopic import BERTopic
            from sentence_transformers import SentenceTransformer
            
            # Initialize BERTopic with car-specific settings
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            self.topic_model = BERTopic(
                embedding_model=self.embedding_model,
                min_topic_size=min_topic_size,
                nr_topics=num_topics,
                calculate_probabilities=True,
                verbose=False
            )
            logger.info(f"Initialized BERTopic with {num_topics} topics")
        except ImportError as e:
            logger.error(f"BERTopic not installed. Install with: pip install bertopic")
            raise
        except Exception as e:
            logger.error(f"Failed to initialize topic modeling: {e}")
            raise

    def __call__(self, texts: List[str]) -> Dict[str, Any]:
        """
        Extract topics from car reviews.

        Args:
            texts: List of texts to analyze.

        Returns:
            Dictionary containing topic modeling results.
        """
        logger.info(f"Extracting topics from {len(texts)} texts")
        try:
            if len(texts) < self.min_topic_size:
                logger.warning(f"Not enough texts ({len(texts)}) for topic modeling. Need at least {self.min_topic_size}")
                return {
                    "topics": [],
                    "topic_labels": [],
                    "topic_words": {},
                    "document_topics": [],
                    "topic_info": "Insufficient data for topic modeling"
                }
            
            # Fit the model and transform documents
            topics, probabilities = self.topic_model.fit_transform(texts)
            
            # Get topic information
            topic_info = self.topic_model.get_topic_info()
            topic_labels = [self.topic_model.get_topic(topic_id) for topic_id in range(len(topic_info))]
            
            # Create topic words dictionary
            topic_words = {}
            for topic_id in range(len(topic_info)):
                if topic_id != -1:  # Exclude outlier topic
                    words = self.topic_model.get_topic(topic_id)
                    topic_words[f"Topic {topic_id}"] = [word for word, score in words[:5]]
            
            # Create document-topic mapping
            document_topics = []
            for i, (text, topic_id, prob) in enumerate(zip(texts, topics, probabilities)):
                topic_name = f"Topic {topic_id}" if topic_id != -1 else "Outlier"
                max_prob = max(prob) if isinstance(prob, list) else prob
                
                document_topics.append({
                    "text": text[:100] + "..." if len(text) > 100 else text,
                    "topic_id": topic_id,
                    "topic_name": topic_name,
                    "probability": float(max_prob),
                    "keywords": topic_words.get(topic_name, [])
                })
            
            results = {
                "topics": topics,
                "topic_labels": topic_labels,
                "topic_words": topic_words,
                "document_topics": document_topics,
                "topic_info": topic_info.to_dict('records') if hasattr(topic_info, 'to_dict') else str(topic_info),
                "num_topics": len(topic_words),
                "outliers": sum(1 for t in topics if t == -1)
            }
            
            logger.info(f"Successfully extracted {len(topic_words)} topics from {len(texts)} documents")
            return results
            
        except Exception as e:
            logger.error(f"Error during topic modeling: {e}")
            # Return a fallback response instead of crashing
            return {
                "topics": [],
                "topic_labels": [],
                "topic_words": {},
                "document_topics": [],
                "topic_info": f"Error: {str(e)}",
                "num_topics": 0,
                "outliers": 0
            }


class AspectSentimentPipeline(BasePipeline):
    """Pipeline for aspect-based sentiment analysis of car reviews."""

    def __init__(self, model_name: str = ASPECT_SENTIMENT_MODEL):
        """
        Initialize the aspect-based sentiment analysis pipeline.

        Args:
            model_name: Name of the pre-trained model to use.
        """
        super().__init__("Aspect-Based Sentiment Analysis")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
            logger.info(f"Loaded aspect-based sentiment model: {model_name}")
        except Exception as e:
            logger.error(f"Failed to load aspect-based sentiment model: {e}")
            raise

        # Define car-related aspects
        self.aspects = [
            "price",
            "performance",
            "comfort",
            "reliability",
            "design",
            "safety",
            "fuel economy",
        ]

    def __call__(self, text: str) -> Dict[str, Dict[str, float]]:
        """
        Analyze aspect-based sentiment of car reviews.

        Args:
            text: Text to analyze.

        Returns:
            Dictionary mapping aspects to sentiment scores.
        """
        logger.info(f"Analyzing aspect-based sentiment (text length: {len(text)})")
        try:
            results = {}

            # Process each aspect
            for aspect in self.aspects:
                # Create input text in the format expected by the model
                input_text = f"{aspect} : {text}"

                # Tokenize input
                inputs = self.tokenizer(
                    input_text, return_tensors="pt", truncation=True, max_length=512
                )

                # Get model predictions
                with torch.no_grad():
                    outputs = self.model(**inputs)

                # Convert logits to probabilities
                probs = torch.nn.functional.softmax(outputs.logits, dim=-1)[0]

                # Map to sentiment labels (model-specific)
                # Assuming 3 classes: negative (0), neutral (1), positive (2)
                sentiment_scores = {
                    "negative": probs[0].item(),
                    "neutral": probs[1].item(),
                    "positive": probs[2].item(),
                }

                results[aspect] = sentiment_scores

            return results
        except Exception as e:
            logger.error(f"Error during aspect-based sentiment analysis: {e}")
            raise
