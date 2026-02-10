# Analyzing Car Reviews with LLMs

An NLP pipeline for analyzing car reviews using HuggingFace transformer models.

## Overview

This project applies multiple NLP techniques to automotive customer reviews: sentiment analysis, text summarization, question answering, English-to-Spanish translation, topic modeling (via BERTopic), aspect-based sentiment analysis, and named entity recognition. Results can be exported in several formats and visualized through static plots or an interactive Plotly dashboard.

## Features

- Sentiment analysis using DistilBERT (SST-2)
- Topic modeling with BERTopic and sentence-transformer embeddings
- English-to-Spanish translation using Helsinki-NLP Opus-MT
- Extractive question answering using MiniLM (SQuAD2)
- Abstractive summarization using T5-small
- Aspect-based sentiment analysis (price, performance, comfort, reliability, design, safety, fuel economy) using DeBERTa-v3
- Named entity recognition for automotive entities
- Multi-format data loading: CSV, JSON, Excel (.xlsx), TSV, TXT
- Multi-format result export: JSON, CSV, Excel, TXT
- Interactive Plotly dashboard and static matplotlib/seaborn charts (bar plots, word clouds)
- Thread-safe model caching with LRU eviction and memory-aware cleanup
- Structured error handling with automatic retries, fallback strategies, and actionable suggestions
- CLI with Rich progress bars and system resource display

## Architecture

| Module | Description |
|---|---|
| `src/pipelines.py` | Core NLP pipeline classes (sentiment, translation, summarization, topic modeling, aspect sentiment) |
| `src/models.py` | Question answering and named entity recognition model wrappers |
| `src/data_processor.py` | Multi-format data loading, validation, preprocessing, and export |
| `src/model_cache.py` | Thread-safe singleton model cache with memory management |
| `src/error_handler.py` | Custom exception hierarchy, retry decorator, system resource validation |
| `src/enhanced_cli.py` | Click-based CLI with Rich output, progress tracking, and input validation |
| `src/visualization.py` | Sentiment/topic/entity bar charts, aspect heatmaps, word clouds, Plotly dashboard |
| `src/config.py` | Centralized configuration with environment variable overrides |
| `src/logger.py` | Logging setup |
| `src/utils.py` | Data loading helpers, metrics calculation, BLEU scoring |
| `main.py` | Original argparse-based CLI entry point |

## Project Structure

```
Analyzing-Car-Reviews-with-LLMs/
├── dataset/
│   ├── car_reviews.csv
│   └── reference_translations.txt
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── data_processor.py
│   ├── enhanced_cli.py
│   ├── error_handler.py
│   ├── logger.py
│   ├── model_cache.py
│   ├── models.py
│   ├── pipelines.py
│   ├── utils.py
│   └── visualization.py
├── tests/
│   ├── conftest.py
│   ├── test_data_processor.py
│   ├── test_enhanced_cli.py
│   ├── test_error_handler.py
│   ├── test_model_cache.py
│   ├── test_models.py
│   └── test_pipelines.py
├── .env.example
├── Dockerfile
├── LICENSE
├── main.py
├── pyproject.toml
├── README.md
└── requirements.txt
```

## Setup

### Prerequisites

- Python 3.10 or later
- pip

Key dependencies: `transformers`, `torch`, `bertopic`, `sentence-transformers`, `plotly`, `matplotlib`, `seaborn`, `wordcloud`, `rich`, `click`, `pandas`, `scikit-learn`. See `requirements.txt` for the full list with pinned versions.

### Installation

```bash
git clone https://github.com/phoenixak/Analyzing-Car-Reviews-with-LLMs.git
cd Analyzing-Car-Reviews-with-LLMs
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Or install as a package (provides the `car-reviews` command):

```bash
pip install -e .
```

### Configuration

Copy `.env.example` to `.env` and edit as needed. Environment variables control model selection, batch size, memory thresholds, visualization settings, export formats, and directory paths. All settings have sensible defaults and work without a `.env` file.

## Usage

### CLI (Enhanced)

The primary interface uses Click and Rich for progress tracking:

```bash
# Run all tasks
python -m src.enhanced_cli --task all --visualize --save-results --verbose

# Sentiment analysis only
python -m src.enhanced_cli --task sentiment --verbose

# Multiple specific tasks
python -m src.enhanced_cli --task sentiment,topic,qa --visualize

# Limit the number of reviews processed
python -m src.enhanced_cli --task sentiment --max-reviews 20

# Show current configuration and system info
python -m src.enhanced_cli --show-config

# Show available models and cache status
python -m src.enhanced_cli --show-models

# Preload models into cache
python -m src.enhanced_cli --preload-models

# Display usage examples
python -m src.enhanced_cli --show-examples
```

If installed as a package, replace `python -m src.enhanced_cli` with `car-reviews`.

### CLI (Classic)

The original argparse interface via `main.py`:

```bash
python main.py --task sentiment --visualize --save-results
python main.py --data-file path/to/reviews.csv --task all
```

Available tasks: `sentiment`, `translation`, `qa`, `summarization`, `topic`, `aspect`, `ner`, `all`.

### Python API

```python
from src.pipelines import SentimentAnalysisPipeline, TopicModelingPipeline
from src.data_processor import EnhancedDataProcessor

# Load data
processor = EnhancedDataProcessor()
reviews, labels = processor.load_data("dataset/car_reviews.csv")

# Sentiment analysis
sentiment = SentimentAnalysisPipeline()
results = sentiment(reviews)

# Topic modeling
topic_pipeline = TopicModelingPipeline(num_topics=5)
topics = topic_pipeline(reviews)

# Export results
processor.export_results({"sentiment": results}, formats=["json", "csv"])
```

## Docker

```bash
docker build -t car-reviews-llm .
docker run -v "$(pwd)/results":/app/results -it car-reviews-llm
```

The default entrypoint runs `python -m src.enhanced_cli --task all --visualize --save-results --verbose`. Override by appending arguments:

```bash
docker run -it car-reviews-llm --task sentiment --max-reviews 10
```

## Testing

```bash
pip install -e ".[dev]"
python -m pytest tests/ -v
```

## License

MIT -- see [LICENSE](LICENSE).
