#!/usr/bin/env python

"""
Main script for analyzing car reviews using Large Language Models.

This script demonstrates various NLP tasks on car reviews:
- Sentiment Analysis: Classify reviews as positive or negative
- Translation: Translate reviews from English to Spanish
- Question Answering: Extract answers from reviews based on questions
- Summarization: Summarize long reviews
- Topic Modeling: Extract topics from reviews
- Aspect-Based Sentiment Analysis: Analyze sentiment for specific aspects
- Named Entity Recognition: Extract entities like car brands and models
"""

import os
import argparse
import pandas as pd
from typing import Dict, List, Any

from src.pipelines import (
    SentimentAnalysisPipeline,
    TranslationPipeline,
    SummarizationPipeline,
    TopicModelingPipeline,
    AspectSentimentPipeline,
)
from src.models import (
    QuestionAnsweringModel,
    NamedEntityRecognitionModel,
)
from src.utils import load_data, calculate_metrics, calculate_bleu_score, save_results
from src.visualization import (
    plot_sentiment_distribution,
    plot_aspect_sentiment,
    plot_topic_distribution,
    plot_named_entities,
    generate_wordcloud,
    create_interactive_dashboard,
)
from src.logger import setup_logger, get_logger
from src.config import DATASET_PATH, REFERENCE_TRANSLATIONS_PATH

# Set up logger
logger = get_logger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Analyze car reviews using Large Language Models"
    )

    parser.add_argument(
        "--data-file",
        type=str,
        default=str(DATASET_PATH),
        help="Path to the car reviews dataset",
    )

    parser.add_argument(
        "--task",
        type=str,
        choices=[
            "sentiment",
            "translation",
            "qa",
            "summarization",
            "topic",
            "aspect",
            "ner",
            "all",
        ],
        default="all",
        help="NLP task to perform",
    )

    parser.add_argument(
        "--visualize", action="store_true", help="Generate visualizations"
    )

    parser.add_argument(
        "--save-results", action="store_true", help="Save results to disk"
    )

    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")

    return parser.parse_args()


def sentiment_analysis(
    reviews: List[str], real_labels: List[str], visualize: bool = False
) -> Dict[str, Any]:
    """
    Perform sentiment analysis on car reviews.

    Args:
        reviews: List of car reviews.
        real_labels: List of true sentiment labels.
        visualize: Whether to generate visualizations.

    Returns:
        Dictionary of sentiment analysis results.
    """
    logger.info("Performing sentiment analysis")

    # Initialize sentiment analysis pipeline
    sentiment_pipeline = SentimentAnalysisPipeline()

    # Analyze sentiment
    predicted_labels = sentiment_pipeline(reviews)

    # Calculate metrics
    metrics = calculate_metrics(real_labels, predicted_labels)

    # Generate visualizations if requested
    if visualize:
        plot_sentiment_distribution(
            predicted_labels,
            title="Predicted Sentiment Distribution",
            filename="sentiment_distribution.png",
        )

    # Return results
    return {
        "task": "sentiment_analysis",
        "metrics": metrics,
        "predictions": predicted_labels,
    }


def translation_analysis(reviews: List[str], visualize: bool = False) -> Dict[str, Any]:
    """
    Translate car reviews from English to Spanish.

    Args:
        reviews: List of car reviews.
        visualize: Whether to generate visualizations.

    Returns:
        Dictionary of translation results.
    """
    logger.info("Performing translation analysis")

    # Initialize translation pipeline
    translation_pipeline = TranslationPipeline()

    # Translate first review
    first_review = reviews[0]
    translated_review = translation_pipeline(first_review)

    # Load reference translations
    try:
        with open(REFERENCE_TRANSLATIONS_PATH, "r") as file:
            lines = file.readlines()
        references = [line.strip() for line in lines]
    except Exception as e:
        logger.error(f"Error loading reference translations: {e}")
        references = []

    # Calculate BLEU score if references are available
    bleu_score = None
    if references:
        bleu_score = calculate_bleu_score(translated_review, references)

    # Return results
    return {
        "task": "translation",
        "original_text": first_review,
        "translated_text": translated_review,
        "references": references,
        "bleu_score": bleu_score,
    }


def question_answering(reviews: List[str], visualize: bool = False) -> Dict[str, Any]:
    """
    Answer questions about car reviews.

    Args:
        reviews: List of car reviews.
        visualize: Whether to generate visualizations.

    Returns:
        Dictionary of question answering results.
    """
    logger.info("Performing question answering")

    # Initialize question answering model
    qa_model = QuestionAnsweringModel()

    # Define questions
    questions = [
        "What did the reviewer like about the car?",
        "What were the negative aspects mentioned?",
        "What features were highlighted in the review?",
    ]

    # Answer questions for a sample review
    context = reviews[1]
    answers = []

    for question in questions:
        answer = qa_model(context, question)
        answers.append({"question": question, "answer": answer})

    # Return results
    return {"task": "question_answering", "context": context, "qa_pairs": answers}


def summarization_analysis(
    reviews: List[str], visualize: bool = False
) -> Dict[str, Any]:
    """
    Summarize car reviews.

    Args:
        reviews: List of car reviews.
        visualize: Whether to generate visualizations.

    Returns:
        Dictionary of summarization results.
    """
    logger.info("Performing summarization analysis")

    # Initialize summarization pipeline
    summarization_pipeline = SummarizationPipeline()

    # Select a long review
    long_review = max(reviews, key=len)

    # Summarize review
    summarized_text = summarization_pipeline(long_review)

    # Calculate compression ratio
    compression_ratio = len(summarized_text) / len(long_review)

    # Return results
    return {
        "task": "summarization",
        "original_text": long_review,
        "summarized_text": summarized_text,
        "original_length": len(long_review),
        "summary_length": len(summarized_text),
        "compression_ratio": compression_ratio,
    }


def topic_modeling(reviews: List[str], visualize: bool = False) -> Dict[str, Any]:
    """
    Extract topics from car reviews.

    Args:
        reviews: List of car reviews.
        visualize: Whether to generate visualizations.

    Returns:
        Dictionary of topic modeling results.
    """
    logger.info("Performing topic modeling")

    # Initialize topic modeling pipeline
    topic_pipeline = TopicModelingPipeline()

    # Extract topics (limit to 20 reviews for better topic modeling)
    topic_results = topic_pipeline(reviews[:20])

    # Generate visualizations if requested
    if visualize and topic_results.get("document_topics"):
        # Create a format compatible with the visualization function
        viz_data = []
        for doc_topic in topic_results["document_topics"]:
            viz_data.append({
                "topics": [{
                    "topic_id": doc_topic["topic_id"],
                    "score": doc_topic["probability"]
                }]
            })
        
        plot_topic_distribution(
            viz_data, title="Topic Distribution", filename="topic_distribution.png"
        )

    # Return results
    return {"task": "topic_modeling", "results": topic_results}


def aspect_sentiment_analysis(
    reviews: List[str], visualize: bool = False
) -> Dict[str, Any]:
    """
    Analyze sentiment for specific aspects of car reviews.

    Args:
        reviews: List of car reviews.
        visualize: Whether to generate visualizations.

    Returns:
        Dictionary of aspect-based sentiment analysis results.
    """
    logger.info("Performing aspect-based sentiment analysis")

    # Initialize aspect sentiment pipeline
    aspect_pipeline = AspectSentimentPipeline()

    # Analyze a sample review
    sample_review = reviews[2]
    aspect_results = aspect_pipeline(sample_review)

    # Generate visualizations if requested
    if visualize:
        plot_aspect_sentiment(
            aspect_results,
            title="Aspect-Based Sentiment Analysis",
            filename="aspect_sentiment.png",
        )

    # Return results
    return {
        "task": "aspect_sentiment_analysis",
        "text": sample_review,
        "aspects": aspect_results,
    }


def named_entity_recognition(
    reviews: List[str], visualize: bool = False
) -> Dict[str, Any]:
    """
    Extract named entities from car reviews.

    Args:
        reviews: List of car reviews.
        visualize: Whether to generate visualizations.

    Returns:
        Dictionary of named entity recognition results.
    """
    logger.info("Performing named entity recognition")

    # Initialize NER model
    ner_model = NamedEntityRecognitionModel()

    # Extract entities from a sample review
    sample_review = reviews[3]
    entities = ner_model(sample_review)

    # Generate visualizations if requested
    if visualize:
        plot_named_entities(
            entities, title="Named Entities", filename="named_entities.png"
        )

    # Return results
    return {
        "task": "named_entity_recognition",
        "text": sample_review,
        "entities": entities,
    }


def main():
    """Main function."""
    # Parse arguments
    args = parse_args()

    # Configure logging
    log_level = "INFO" if args.verbose else "WARNING"
    setup_logger(log_level=log_level)

    logger.info("Starting car reviews analysis")

    # Load data
    reviews, real_labels = load_data(args.data_file)
    logger.info(f"Loaded {len(reviews)} reviews from {args.data_file}")

    # Store results
    results = {}

    # Perform requested tasks
    if args.task in ["sentiment", "all"]:
        results["sentiment"] = sentiment_analysis(reviews, real_labels, args.visualize)

    if args.task in ["translation", "all"]:
        results["translation"] = translation_analysis(reviews, args.visualize)

    if args.task in ["qa", "all"]:
        results["qa"] = question_answering(reviews, args.visualize)

    if args.task in ["summarization", "all"]:
        results["summarization"] = summarization_analysis(reviews, args.visualize)

    if args.task in ["topic", "all"]:
        results["topic"] = topic_modeling(reviews, args.visualize)

    if args.task in ["aspect", "all"]:
        results["aspect"] = aspect_sentiment_analysis(reviews, args.visualize)

    if args.task in ["ner", "all"]:
        results["ner"] = named_entity_recognition(reviews, args.visualize)

    # Generate word cloud if visualizations are enabled
    if args.visualize:
        generate_wordcloud(
            reviews, title="Car Reviews Word Cloud", filename="word_cloud.png"
        )

        # Create interactive dashboard
        create_interactive_dashboard(
            results, title="Car Reviews Analysis Dashboard", filename="dashboard.html"
        )

    # Save results if requested
    if args.save_results:
        save_path = save_results(results, "car_reviews_analysis_results.json")
        logger.info(f"Results saved to {save_path}")

    logger.info("Analysis completed")


if __name__ == "__main__":
    main()
