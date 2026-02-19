# Analyzing Car Reviews with LLMs

An NLP pipeline for analyzing car reviews using HuggingFace transformer models.

**Live demo:** [Hugging Face Space](https://huggingface.co/spaces/Phoenixak99/car-reviews-analyzer)

## Overview

This project applies seven NLP techniques to automotive customer reviews: sentiment analysis, text summarization, question answering, English-to-Spanish translation, topic modeling (via BERTopic), aspect-based sentiment analysis, and named entity recognition. Results can be exported in multiple formats and visualized through static plots or an interactive Plotly dashboard.

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
- Click-based CLI with Rich progress bars and system resource display
- Sentiment evaluation harness with accuracy, precision, recall, and F1 reporting

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
| `src/evaluation.py` | Sentiment model evaluation on a labeled test set |
| `src/config.py` | Centralized configuration with environment variable overrides |
| `src/logger.py` | Logging setup |
| `src/utils.py` | Data loading helpers, metrics calculation, BLEU scoring |
| `main.py` | Argparse-based CLI entry point |
| `scripts/run_evaluation.py` | Standalone script to run and save sentiment evaluation metrics |
| `hf_space/app.py` | Gradio web demo (mirrors CLI features in a browser UI) |

## Project Structure

```
Analyzing-Car-Reviews-with-LLMs/
├── dataset/
│   ├── car_reviews.csv
│   └── reference_translations.txt
├── hf_space/
│   ├── app.py
│   ├── requirements.txt
│   └── sample_reviews.txt
├── scripts/
│   └── run_evaluation.py
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── data_processor.py
│   ├── enhanced_cli.py
│   ├── error_handler.py
│   ├── evaluation.py
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

> **Note on Python versions:** `requirements.txt` pins specific versions tested on Python 3.10.
> On Python 3.11+ install via `pip install -e .` instead (the `pyproject.toml` uses flexible bounds
> and works with current package versions). Both methods are described below.

Key dependencies: `transformers`, `torch`, `bertopic`, `sentence-transformers`, `plotly`, `matplotlib`, `seaborn`, `wordcloud`, `rich`, `click`, `pandas`, `scikit-learn`. See `requirements.txt` for the pinned list.

### Installation

```bash
git clone https://github.com/phoenixak/Analyzing-Car-Reviews-with-LLMs.git
cd Analyzing-Car-Reviews-with-LLMs
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
```

**Option A — pinned versions (Python 3.10 recommended):**

```bash
pip install -r requirements.txt
```

**Option B — flexible versions (Python 3.11/3.12+):**

```bash
pip install -e .
```

Option B also installs the `car-reviews` command (see [CLI section](#cli-enhanced)).

### Configuration

Copy `.env.example` to `.env` and edit as needed:

```bash
cp .env.example .env   # Windows: copy .env.example .env
```

Environment variables control model selection, batch size, memory thresholds, visualization settings, export formats, and directory paths. All settings have sensible defaults — the project runs without a `.env` file.

The most useful variable to set is `HF_HOME`, which controls where HuggingFace caches downloaded models:

```bash
HF_HOME=/path/to/your/model/cache   # defaults to ~/.cache/huggingface
```

> **First-run model downloads:** the first time each NLP task runs, its model is downloaded from
> HuggingFace (total ~3-5 GB across all tasks). Subsequent runs load from the local cache and are
> much faster. Download progress is shown automatically; you can pre-download with `--preload-models`.

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

# Evaluate the sentiment model and print accuracy/F1 metrics
python -m src.enhanced_cli --evaluate

# Show current configuration and system info
python -m src.enhanced_cli --show-config

# Show available models and cache status
python -m src.enhanced_cli --show-models

# Preload all models into cache before running tasks
python -m src.enhanced_cli --preload-models

# Display usage examples
python -m src.enhanced_cli --show-examples
```

If installed as a package (`pip install -e .`), replace `python -m src.enhanced_cli` with `car-reviews`.

### CLI (Classic)

The original argparse interface via `main.py`:

```bash
python main.py --task sentiment --visualize --save-results
python main.py --data-file path/to/reviews.csv --task all
python main.py --task evaluate   # runs sentiment evaluation and prints metrics
```

Available tasks: `sentiment`, `translation`, `qa`, `summarization`, `topic`, `aspect`, `ner`, `evaluate`, `all`.

### Evaluation Script

Run the standalone evaluation script to compute and save sentiment model metrics:

```bash
python scripts/run_evaluation.py

# Optional flags
python scripts/run_evaluation.py --output-dir results/eval
python scripts/run_evaluation.py --model distilbert-base-uncased-finetuned-sst-2-english
```

This classifies a built-in set of 50 labeled car review sentences and prints accuracy, precision, recall, and F1. Results are saved to `results/evaluation_metrics.json`.

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

## Results & Output

All output is written to the `results/` directory (created automatically on first run):

| File | Description |
|---|---|
| `car_reviews_analysis_results.json` | Full analysis results in JSON |
| `car_reviews_analysis_results.csv` | Flattened predictions and metrics in CSV |
| `car_reviews_analysis_results.xlsx` | Excel workbook with per-task sheets |
| `evaluation_metrics.json` | Sentiment evaluation metrics (accuracy, F1, etc.) |
| `dashboard.html` | Interactive Plotly dashboard (open in browser) |
| `sentiment_distribution.png` | Sentiment bar chart |
| `topic_distribution.png` | Topic distribution chart |
| `aspect_sentiment.png` | Aspect sentiment heatmap |
| `named_entities.png` | Named entity frequency chart |
| `word_cloud.png` | Word cloud of all review text |

Log output is written to `logs/car_reviews_analysis.log`.

## Gradio Demo (hf_space/)

The `hf_space/` directory contains a standalone Gradio app that exposes all seven NLP tasks in a browser UI. It is deployed to the [live Hugging Face Space](https://huggingface.co/spaces/Phoenixak99/car-reviews-analyzer) and can also be run locally:

```bash
pip install -r hf_space/requirements.txt
python hf_space/app.py
```

The app will be available at `http://localhost:7860`.

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

The test suite has 115 tests and runs entirely with mocked models — no model downloads required.

## License

MIT -- see [LICENSE](LICENSE).
