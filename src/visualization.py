"""
Visualization functions for the car reviews analysis project.

This module provides functions for visualizing analysis results.
"""

import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union
from wordcloud import WordCloud
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter

from src.logger import get_logger
from src.config import RESULTS_DIR, FIGSIZE, DPI

# Set up logger
logger = get_logger(__name__)


def plot_sentiment_distribution(
    sentiment_results: List[Dict[str, Any]],
    title: str = "Sentiment Distribution",
    filename: Optional[str] = None,
) -> None:
    """
    Plot the distribution of sentiment labels.

    Args:
        sentiment_results: List of sentiment analysis results.
        title: Title for the plot.
        filename: If provided, save the plot to this file.
    """
    logger.info(f"Plotting sentiment distribution: {title}")
    try:
        # Extract labels
        labels = [result["label"] for result in sentiment_results]

        # Count sentiment labels
        label_counts = pd.Series(labels).value_counts()

        # Create plot
        plt.figure(figsize=FIGSIZE)
        ax = sns.barplot(x=label_counts.index, y=label_counts.values)

        # Add count labels on top of bars
        for i, count in enumerate(label_counts.values):
            ax.text(i, count + 0.1, str(count), ha="center")

        plt.title(title)
        plt.xlabel("Sentiment")
        plt.ylabel("Count")

        # Save if filename is provided
        if filename:
            os.makedirs(RESULTS_DIR, exist_ok=True)
            file_path = os.path.join(RESULTS_DIR, filename)
            plt.savefig(file_path, dpi=DPI, bbox_inches="tight")
            logger.info(f"Plot saved to {file_path}")

        plt.close()
    except Exception as e:
        logger.error(f"Error plotting sentiment distribution: {e}")
        raise


def plot_aspect_sentiment(
    aspect_results: Dict[str, Dict[str, float]],
    title: str = "Aspect-Based Sentiment Analysis",
    filename: Optional[str] = None,
) -> None:
    """
    Plot the sentiment scores for different aspects.

    Args:
        aspect_results: Dictionary mapping aspects to sentiment scores.
        title: Title for the plot.
        filename: If provided, save the plot to this file.
    """
    logger.info(f"Plotting aspect sentiment: {title}")
    try:
        # Extract data
        aspects = list(aspect_results.keys())
        positive_scores = [aspect_results[aspect]["positive"] for aspect in aspects]
        neutral_scores = [aspect_results[aspect]["neutral"] for aspect in aspects]
        negative_scores = [aspect_results[aspect]["negative"] for aspect in aspects]

        # Create DataFrame
        df = pd.DataFrame(
            {
                "Aspect": aspects,
                "Positive": positive_scores,
                "Neutral": neutral_scores,
                "Negative": negative_scores,
            }
        )

        # Melt DataFrame for easier plotting
        df_melted = df.melt(
            id_vars=["Aspect"], var_name="Sentiment", value_name="Score"
        )

        # Create plot
        plt.figure(figsize=FIGSIZE)
        ax = sns.barplot(x="Aspect", y="Score", hue="Sentiment", data=df_melted)

        plt.title(title)
        plt.xlabel("Aspect")
        plt.ylabel("Score")
        plt.xticks(rotation=45)
        plt.legend(title="Sentiment")
        plt.tight_layout()

        # Save if filename is provided
        if filename:
            os.makedirs(RESULTS_DIR, exist_ok=True)
            file_path = os.path.join(RESULTS_DIR, filename)
            plt.savefig(file_path, dpi=DPI, bbox_inches="tight")
            logger.info(f"Plot saved to {file_path}")

        plt.close()
    except Exception as e:
        logger.error(f"Error plotting aspect sentiment: {e}")
        raise


def plot_topic_distribution(
    topic_results: List[Dict[str, Any]],
    title: str = "Topic Distribution",
    filename: Optional[str] = None,
) -> None:
    """
    Plot the distribution of topics.

    Args:
        topic_results: List of topic modeling results.
        title: Title for the plot.
        filename: If provided, save the plot to this file.
    """
    logger.info(f"Plotting topic distribution: {title}")
    try:
        # Extract topics and scores
        all_topics = []
        for result in topic_results:
            for topic in result["topics"]:
                all_topics.append((topic["topic_id"], topic["score"]))

        # Count topics
        topic_counts = Counter([topic[0] for topic in all_topics])

        # Create plot
        plt.figure(figsize=FIGSIZE)
        topics = list(topic_counts.keys())
        counts = list(topic_counts.values())

        ax = sns.barplot(x=[f"Topic {t}" for t in topics], y=counts)

        # Add count labels on top of bars
        for i, count in enumerate(counts):
            ax.text(i, count + 0.1, str(count), ha="center")

        plt.title(title)
        plt.xlabel("Topic")
        plt.ylabel("Count")
        plt.xticks(rotation=45)
        plt.tight_layout()

        # Save if filename is provided
        if filename:
            os.makedirs(RESULTS_DIR, exist_ok=True)
            file_path = os.path.join(RESULTS_DIR, filename)
            plt.savefig(file_path, dpi=DPI, bbox_inches="tight")
            logger.info(f"Plot saved to {file_path}")

        plt.close()
    except Exception as e:
        logger.error(f"Error plotting topic distribution: {e}")
        raise


def plot_named_entities(
    entities: List[Dict[str, Any]],
    title: str = "Named Entities",
    filename: Optional[str] = None,
) -> None:
    """
    Plot the distribution of named entities.

    Args:
        entities: List of named entities.
        title: Title for the plot.
        filename: If provided, save the plot to this file.
    """
    logger.info(f"Plotting named entities: {title}")
    try:
        # Count entity types
        entity_types = [entity["entity"] for entity in entities]
        entity_counts = pd.Series(entity_types).value_counts()

        # Create plot
        plt.figure(figsize=FIGSIZE)
        ax = sns.barplot(x=entity_counts.index, y=entity_counts.values)

        # Add count labels on top of bars
        for i, count in enumerate(entity_counts.values):
            ax.text(i, count + 0.1, str(count), ha="center")

        plt.title(title)
        plt.xlabel("Entity Type")
        plt.ylabel("Count")
        plt.xticks(rotation=45)
        plt.tight_layout()

        # Save if filename is provided
        if filename:
            os.makedirs(RESULTS_DIR, exist_ok=True)
            file_path = os.path.join(RESULTS_DIR, filename)
            plt.savefig(file_path, dpi=DPI, bbox_inches="tight")
            logger.info(f"Plot saved to {file_path}")

        plt.close()
    except Exception as e:
        logger.error(f"Error plotting named entities: {e}")
        raise


def create_interactive_dashboard(
    results: Dict[str, Any],
    title: str = "Car Reviews Analysis Dashboard",
    filename: Optional[str] = "dashboard.html",
) -> None:
    """
    Create an interactive dashboard for car reviews analysis.

    Args:
        results: Dictionary of analysis results.
        title: Title for the dashboard.
        filename: If provided, save the dashboard to this file.
    """
    logger.info(f"Creating interactive dashboard: {title}")
    try:
        from plotly.subplots import make_subplots
        
        # Create subplots for multiple visualizations
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Sentiment Distribution', 'Aspect Sentiment', 'Topic Distribution', 'Summary Stats'),
            specs=[[{"type": "bar"}, {"type": "bar"}],
                   [{"type": "bar"}, {"type": "table"}]]
        )

        # Add sentiment distribution (fix key name mismatch)
        sentiment_data = results.get("sentiment", {})
        if sentiment_data and "predictions" in sentiment_data:
            sentiment_labels = [pred["label"] for pred in sentiment_data["predictions"]]
            sentiment_counts = pd.Series(sentiment_labels).value_counts()

            fig.add_trace(
                go.Bar(
                    x=sentiment_counts.index,
                    y=sentiment_counts.values,
                    name="Sentiment Distribution",
                    marker_color=[
                        "#FF6B6B" if label == "NEGATIVE" else "#4ECDC4"
                        for label in sentiment_counts.index
                    ],
                ),
                row=1, col=1
            )

        # Add aspect sentiment (fix key name mismatch)
        aspect_data = results.get("aspect", {})
        if aspect_data and "aspects" in aspect_data:
            aspects = list(aspect_data["aspects"].keys())
            positive_scores = [aspect_data["aspects"][aspect]["positive"] for aspect in aspects]
            negative_scores = [aspect_data["aspects"][aspect]["negative"] for aspect in aspects]

            fig.add_trace(
                go.Bar(
                    x=aspects,
                    y=positive_scores,
                    name="Positive",
                    marker_color="#4ECDC4",
                ),
                row=1, col=2
            )
            
            fig.add_trace(
                go.Bar(
                    x=aspects,
                    y=negative_scores,
                    name="Negative",
                    marker_color="#FF6B6B",
                ),
                row=1, col=2
            )

        # Add topic distribution
        topic_data = results.get("topic", {})
        if topic_data and "results" in topic_data and "topic_words" in topic_data["results"]:
            topic_words = topic_data["results"]["topic_words"]
            topics = list(topic_words.keys())
            topic_counts = [len(topic_words[topic]) for topic in topics]

            fig.add_trace(
                go.Bar(
                    x=topics,
                    y=topic_counts,
                    name="Topic Keywords",
                    marker_color="#45B7D1",
                ),
                row=2, col=1
            )

        # Add summary statistics table
        summary_stats = []
        for task_name, task_data in results.items():
            if isinstance(task_data, dict):
                if "metrics" in task_data:
                    metrics = task_data["metrics"]
                    summary_stats.append([
                        task_name.title(),
                        f"{metrics.get('accuracy', 'N/A'):.3f}" if isinstance(metrics.get('accuracy'), float) else 'N/A',
                        f"{metrics.get('f1', 'N/A'):.3f}" if isinstance(metrics.get('f1'), float) else 'N/A'
                    ])
                elif task_name == "topic" and "num_topics" in task_data.get("results", {}):
                    summary_stats.append([
                        task_name.title(),
                        str(task_data["results"].get("num_topics", "N/A")),
                        str(task_data["results"].get("outliers", "N/A"))
                    ])

        if summary_stats:
            fig.add_trace(
                go.Table(
                    header=dict(values=['Task', 'Accuracy/Count', 'F1/Outliers'],
                              fill_color='#f0f0f0',
                              align='left'),
                    cells=dict(values=list(zip(*summary_stats)) if summary_stats else [[], [], []],
                             fill_color='white',
                             align='left')
                ),
                row=2, col=2
            )

        # Update layout
        fig.update_layout(
            title=title,
            showlegend=True,
            template="plotly_white",
            height=800,
        )

        # Save if filename is provided
        if filename:
            os.makedirs(RESULTS_DIR, exist_ok=True)
            file_path = os.path.join(RESULTS_DIR, filename)
            fig.write_html(file_path)
            logger.info(f"Enhanced dashboard saved to {file_path}")
            
    except Exception as e:
        logger.error(f"Error creating interactive dashboard: {e}")
        # Create a simple fallback dashboard
        try:
            simple_fig = go.Figure()
            simple_fig.add_annotation(
                text=f"Dashboard creation failed: {str(e)}<br>Results available: {list(results.keys())}",
                xref="paper", yref="paper",
                x=0.5, y=0.5, 
                showarrow=False,
                font=dict(size=16)
            )
            simple_fig.update_layout(title="Car Reviews Analysis - Error Dashboard")
            
            if filename:
                os.makedirs(RESULTS_DIR, exist_ok=True)
                file_path = os.path.join(RESULTS_DIR, filename)
                simple_fig.write_html(file_path)
                logger.info(f"Fallback dashboard saved to {file_path}")
        except Exception as fallback_error:
            logger.error(f"Failed to create fallback dashboard: {fallback_error}")
            raise


def generate_wordcloud(
    texts: List[str], title: str = "Word Cloud", filename: Optional[str] = None
) -> None:
    """
    Generate a word cloud from texts.

    Args:
        texts: List of texts.
        title: Title for the word cloud.
        filename: If provided, save the word cloud to this file.
    """
    logger.info(f"Generating word cloud: {title}")
    try:
        # Combine all texts
        text = " ".join(texts)

        # Create word cloud
        wordcloud = WordCloud(
            width=800,
            height=400,
            background_color="white",
            max_words=200,
            contour_width=3,
        ).generate(text)

        # Display word cloud
        plt.figure(figsize=FIGSIZE)
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        plt.title(title)

        # Save if filename is provided
        if filename:
            os.makedirs(RESULTS_DIR, exist_ok=True)
            file_path = os.path.join(RESULTS_DIR, filename)
            plt.savefig(file_path, dpi=DPI, bbox_inches="tight")
            logger.info(f"Word cloud saved to {file_path}")

        plt.close()
    except Exception as e:
        logger.error(f"Error generating word cloud: {e}")
        raise
