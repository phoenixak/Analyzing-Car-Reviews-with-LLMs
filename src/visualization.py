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


def _ensure_results_dir():
    """Ensure the results directory exists."""
    os.makedirs(RESULTS_DIR, exist_ok=True)


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
            _ensure_results_dir()
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
        df_melted = df.melt(id_vars=["Aspect"], var_name="Sentiment", value_name="Score")

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
            _ensure_results_dir()
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
            _ensure_results_dir()
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
            _ensure_results_dir()
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
            rows=2,
            cols=2,
            subplot_titles=(
                "Sentiment Distribution",
                "Aspect Sentiment",
                "Topic Distribution",
                "Summary Stats",
            ),
            specs=[
                [{"type": "bar"}, {"type": "bar"}],
                [{"type": "bar"}, {"type": "table"}],
            ],
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
                row=1,
                col=1,
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
                row=1,
                col=2,
            )

            fig.add_trace(
                go.Bar(
                    x=aspects,
                    y=negative_scores,
                    name="Negative",
                    marker_color="#FF6B6B",
                ),
                row=1,
                col=2,
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
                row=2,
                col=1,
            )

        # Add summary statistics table
        summary_stats = []
        for task_name, task_data in results.items():
            if isinstance(task_data, dict):
                if "metrics" in task_data:
                    metrics = task_data["metrics"]
                    summary_stats.append(
                        [
                            task_name.title(),
                            f"{metrics.get('accuracy', 'N/A'):.3f}"
                            if isinstance(metrics.get("accuracy"), float)
                            else "N/A",
                            f"{metrics.get('f1', 'N/A'):.3f}"
                            if isinstance(metrics.get("f1"), float)
                            else "N/A",
                        ]
                    )
                elif task_name == "topic" and "num_topics" in task_data.get("results", {}):
                    summary_stats.append(
                        [
                            task_name.title(),
                            str(task_data["results"].get("num_topics", "N/A")),
                            str(task_data["results"].get("outliers", "N/A")),
                        ]
                    )

        if summary_stats:
            fig.add_trace(
                go.Table(
                    header=dict(
                        values=["Task", "Accuracy/Count", "F1/Outliers"],
                        fill_color="#f0f0f0",
                        align="left",
                    ),
                    cells=dict(
                        values=list(zip(*summary_stats)) if summary_stats else [[], [], []],
                        fill_color="white",
                        align="left",
                    ),
                ),
                row=2,
                col=2,
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
            _ensure_results_dir()
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
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5,
                showarrow=False,
                font=dict(size=16),
            )
            simple_fig.update_layout(title="Car Reviews Analysis - Error Dashboard")

            if filename:
                _ensure_results_dir()
                file_path = os.path.join(RESULTS_DIR, filename)
                simple_fig.write_html(file_path)
                logger.info(f"Fallback dashboard saved to {file_path}")
        except Exception as fallback_error:
            logger.error(f"Failed to create fallback dashboard: {fallback_error}")
            raise


def create_topic_exploration_dashboard(
    topic_results: Dict[str, Any],
    output_dir: Optional[str] = None,
) -> None:
    """
    Create an interactive Plotly dashboard for exploring BERTopic results.

    Displays a 2x2 grid of panels:
    - Panel 1: Topic word importance (horizontal bars, dropdown to switch topics)
    - Panel 2: Topic size distribution (documents per topic)
    - Panel 3: Topic word treemap (words sized by weight)
    - Panel 4: Document-topic probability distribution

    Args:
        topic_results: Dictionary of topic modeling results from TopicModelingPipeline.
            Expected keys: topic_words, topic_labels, document_topics, topics.
        output_dir: Directory to save the dashboard HTML. Defaults to RESULTS_DIR.
    """
    logger.info("Creating topic exploration dashboard")

    if not topic_results or not topic_results.get("topic_words"):
        logger.warning("No topic data available for topic exploration dashboard")
        return

    try:
        from plotly.subplots import make_subplots

        topic_words = topic_results.get("topic_words", {})
        topic_labels_data = topic_results.get("topic_labels", [])
        document_topics = topic_results.get("document_topics", [])
        topics_list = topic_results.get("topics", [])

        if not topic_words:
            logger.warning("No topics found; skipping topic exploration dashboard")
            return

        # Build word importance data per topic: topic_name -> [(word, score), ...]
        topic_word_data: Dict[str, List[tuple]] = {}
        for topic_name, words in topic_words.items():
            idx = int(topic_name.split()[-1])
            if idx < len(topic_labels_data) and isinstance(topic_labels_data[idx], list):
                topic_word_data[topic_name] = topic_labels_data[idx]
            else:
                # Fallback: assign uniform weights when score data is missing
                n = max(len(words), 1)
                topic_word_data[topic_name] = [(w, 1.0 / n) for w in words]

        topic_names = list(topic_word_data.keys())

        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=(
                "Topic Word Importance",
                "Topic Size Distribution",
                "Topic Word Treemap",
                "Document-Topic Probability Distribution",
            ),
            specs=[
                [{"type": "bar"}, {"type": "bar"}],
                [{"type": "treemap"}, {"type": "histogram"}],
            ],
        )

        # Panel 1: Word importance per topic with dropdown to switch topics
        for i, topic_name in enumerate(topic_names):
            words_scores = topic_word_data[topic_name]
            words = [ws[0] for ws in words_scores]
            scores = [abs(ws[1]) for ws in words_scores]
            fig.add_trace(
                go.Bar(
                    y=words,
                    x=scores,
                    orientation="h",
                    name=topic_name,
                    marker_color="#45B7D1",
                    visible=(i == 0),
                    showlegend=False,
                ),
                row=1,
                col=1,
            )

        # Panel 2: Topic size distribution (documents per topic)
        topic_id_counts: Counter = Counter()
        for t in topics_list:
            label = f"Topic {t}" if t != -1 else "Outlier"
            topic_id_counts[label] += 1

        size_labels = sorted(topic_id_counts.keys())
        size_values = [topic_id_counts[lbl] for lbl in size_labels]
        size_colors = ["#FF6B6B" if lbl == "Outlier" else "#4ECDC4" for lbl in size_labels]

        fig.add_trace(
            go.Bar(
                x=size_labels,
                y=size_values,
                name="Documents per Topic",
                marker_color=size_colors,
                showlegend=False,
            ),
            row=1,
            col=2,
        )

        # Panel 3: Treemap of topic words sized by weight
        treemap_ids = ["root"]
        treemap_labels = ["Topics"]
        treemap_parents = [""]
        treemap_values = [0]

        for topic_name, words_scores in topic_word_data.items():
            topic_id = topic_name.replace(" ", "_")
            treemap_ids.append(topic_id)
            treemap_labels.append(topic_name)
            treemap_parents.append("root")
            treemap_values.append(0)

            for word, score in words_scores:
                word_id = f"{topic_id}_{word}"
                treemap_ids.append(word_id)
                treemap_labels.append(word)
                treemap_parents.append(topic_id)
                treemap_values.append(abs(score))

        fig.add_trace(
            go.Treemap(
                ids=treemap_ids,
                labels=treemap_labels,
                parents=treemap_parents,
                values=treemap_values,
                textinfo="label+percent parent",
                marker=dict(colorscale="Teal"),
            ),
            row=2,
            col=1,
        )

        # Panel 4: Document-topic probability distribution
        if document_topics:
            probabilities = [dt["probability"] for dt in document_topics]
            fig.add_trace(
                go.Histogram(
                    x=probabilities,
                    nbinsx=20,
                    name="Probability",
                    marker_color="#FF6B6B",
                    showlegend=False,
                ),
                row=2,
                col=2,
            )

        # Dropdown buttons to switch between topics in Panel 1
        num_topic_traces = len(topic_names)
        total_traces = len(fig.data)
        num_other_traces = total_traces - num_topic_traces

        buttons = []
        for i, topic_name in enumerate(topic_names):
            visibility = [False] * num_topic_traces + [True] * num_other_traces
            visibility[i] = True
            buttons.append(
                dict(
                    label=topic_name,
                    method="update",
                    args=[{"visible": visibility}],
                )
            )

        fig.update_layout(
            updatemenus=[
                dict(
                    active=0,
                    buttons=buttons,
                    x=0.0,
                    xanchor="left",
                    y=1.15,
                    yanchor="top",
                    showactive=True,
                )
            ],
            title="Topic Exploration Dashboard",
            template="plotly_white",
            height=900,
            showlegend=False,
        )

        fig.update_xaxes(title_text="Weight", row=1, col=1)
        fig.update_yaxes(title_text="Word", row=1, col=1)
        fig.update_xaxes(title_text="Topic", row=1, col=2)
        fig.update_yaxes(title_text="Document Count", row=1, col=2)
        fig.update_xaxes(title_text="Probability", row=2, col=2)
        fig.update_yaxes(title_text="Count", row=2, col=2)

        # Save dashboard
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            file_path = os.path.join(output_dir, "topic_exploration_dashboard.html")
        else:
            _ensure_results_dir()
            file_path = os.path.join(RESULTS_DIR, "topic_exploration_dashboard.html")

        fig.write_html(file_path)
        logger.info(f"Topic exploration dashboard saved to {file_path}")

    except Exception as e:
        logger.error(f"Error creating topic exploration dashboard: {e}")
        raise


def create_entity_analysis_dashboard(
    ner_results: List[Dict[str, Any]],
    output_dir: Optional[str] = None,
) -> None:
    """
    Create an interactive Plotly dashboard for exploring NER results.

    Displays a 2x2 grid of panels:
    - Panel 1: Entity type distribution (PER, ORG, LOC, MISC counts)
    - Panel 2: Top entities by frequency (most common entity words)
    - Panel 3: Entity confidence distribution (histogram of scores)
    - Panel 4: Entity type co-occurrence heatmap

    Args:
        ner_results: List of entity dicts from NamedEntityRecognitionModel.
            Each dict has keys: word, entity, score, start, end.
        output_dir: Directory to save the dashboard HTML. Defaults to RESULTS_DIR.
    """
    logger.info("Creating entity analysis dashboard")

    if not ner_results:
        logger.warning("No NER data available for entity analysis dashboard")
        return

    try:
        from plotly.subplots import make_subplots

        entity_types = [e["entity"] for e in ner_results]
        entity_words = [e["word"] for e in ner_results]
        entity_scores = [e["score"] for e in ner_results]

        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=(
                "Entity Type Distribution",
                "Top Entities by Frequency",
                "Entity Confidence Distribution",
                "Entity Type Co-occurrence",
            ),
            specs=[
                [{"type": "bar"}, {"type": "bar"}],
                [{"type": "histogram"}, {"type": "heatmap"}],
            ],
        )

        # Panel 1: Entity type distribution
        type_counts = Counter(entity_types)
        type_labels = sorted(type_counts.keys())
        type_values = [type_counts[t] for t in type_labels]
        type_colors = {
            "PER": "#FF6B6B",
            "ORG": "#4ECDC4",
            "LOC": "#45B7D1",
            "MISC": "#FFA07A",
        }
        colors = [type_colors.get(t, "#95A5A6") for t in type_labels]

        fig.add_trace(
            go.Bar(
                x=type_labels,
                y=type_values,
                name="Entity Types",
                marker_color=colors,
                showlegend=False,
            ),
            row=1,
            col=1,
        )

        # Panel 2: Top entities by frequency (horizontal bar chart, top 15)
        word_counts = Counter(entity_words)
        top_entities = word_counts.most_common(15)
        top_words = [e[0] for e in top_entities][::-1]
        top_freqs = [e[1] for e in top_entities][::-1]

        fig.add_trace(
            go.Bar(
                y=top_words,
                x=top_freqs,
                orientation="h",
                name="Top Entities",
                marker_color="#45B7D1",
                showlegend=False,
            ),
            row=1,
            col=2,
        )

        # Panel 3: Entity confidence distribution
        fig.add_trace(
            go.Histogram(
                x=entity_scores,
                nbinsx=20,
                name="Confidence",
                marker_color="#4ECDC4",
                showlegend=False,
            ),
            row=2,
            col=1,
        )

        # Panel 4: Entity type co-occurrence heatmap
        # Co-occurrence strength: for each pair of types present together,
        # use min(count_a, count_b) as the overlap metric; diagonal = own count.
        unique_types = sorted(set(entity_types))
        n_types = len(unique_types)
        co_matrix = np.zeros((n_types, n_types), dtype=int)

        for i, type_a in enumerate(unique_types):
            count_a = type_counts[type_a]
            for j, type_b in enumerate(unique_types):
                count_b = type_counts[type_b]
                if i == j:
                    co_matrix[i][j] = count_a
                else:
                    co_matrix[i][j] = min(count_a, count_b)

        fig.add_trace(
            go.Heatmap(
                z=co_matrix.tolist(),
                x=unique_types,
                y=unique_types,
                colorscale="Teal",
                showscale=True,
                name="Co-occurrence",
            ),
            row=2,
            col=2,
        )

        fig.update_layout(
            title="Entity Analysis Dashboard",
            template="plotly_white",
            height=900,
            showlegend=False,
        )

        fig.update_xaxes(title_text="Entity Type", row=1, col=1)
        fig.update_yaxes(title_text="Count", row=1, col=1)
        fig.update_xaxes(title_text="Frequency", row=1, col=2)
        fig.update_yaxes(title_text="Entity", row=1, col=2)
        fig.update_xaxes(title_text="Confidence Score", row=2, col=1)
        fig.update_yaxes(title_text="Count", row=2, col=1)

        # Save dashboard
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            file_path = os.path.join(output_dir, "entity_analysis_dashboard.html")
        else:
            _ensure_results_dir()
            file_path = os.path.join(RESULTS_DIR, "entity_analysis_dashboard.html")

        fig.write_html(file_path)
        logger.info(f"Entity analysis dashboard saved to {file_path}")

    except Exception as e:
        logger.error(f"Error creating entity analysis dashboard: {e}")
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
            _ensure_results_dir()
            file_path = os.path.join(RESULTS_DIR, filename)
            plt.savefig(file_path, dpi=DPI, bbox_inches="tight")
            logger.info(f"Word cloud saved to {file_path}")

        plt.close()
    except Exception as e:
        logger.error(f"Error generating word cloud: {e}")
        raise
