"""
Enhanced configuration settings for the car reviews analysis project.
Supports environment variables and flexible model selection.
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional

def get_env_or_default(key: str, default: Any, type_func: callable = str) -> Any:
    """Get environment variable with default value and type conversion."""
    value = os.getenv(key, default)
    try:
        return type_func(value) if value != default else default
    except (ValueError, TypeError):
        return default

# Project paths
PROJECT_ROOT = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATASET_DIR = PROJECT_ROOT / get_env_or_default("DATASET_DIR", "dataset")
RESULTS_DIR = PROJECT_ROOT / get_env_or_default("RESULTS_DIR", "results")
MODELS_DIR = PROJECT_ROOT / get_env_or_default("MODELS_DIR", "models")
CACHE_DIR = PROJECT_ROOT / get_env_or_default("CACHE_DIR", ".cache")

# Create directories if they don't exist
for directory in [RESULTS_DIR, MODELS_DIR, CACHE_DIR]:
    os.makedirs(directory, exist_ok=True)

# Data settings
DATASET_PATH = DATASET_DIR / get_env_or_default("DATASET_FILE", "car_reviews.csv")
REFERENCE_TRANSLATIONS_PATH = DATASET_DIR / get_env_or_default("REFERENCE_FILE", "reference_translations.txt")

# Model settings (configurable via environment variables)
SENTIMENT_MODEL = get_env_or_default("SENTIMENT_MODEL", "distilbert-base-uncased-finetuned-sst-2-english")
TRANSLATION_MODEL = get_env_or_default("TRANSLATION_MODEL", "Helsinki-NLP/opus-mt-en-es")
SUMMARIZATION_MODEL = get_env_or_default("SUMMARIZATION_MODEL", "cnicu/t5-small-booksum")
QA_MODEL = get_env_or_default("QA_MODEL", "deepset/minilm-uncased-squad2")
TOPIC_MODEL = get_env_or_default("TOPIC_MODEL", "all-MiniLM-L6-v2")  # Updated for BERTopic
ASPECT_SENTIMENT_MODEL = get_env_or_default("ASPECT_MODEL", "yangheng/deberta-v3-base-absa-v1.1")

# Performance settings
BATCH_SIZE = get_env_or_default("BATCH_SIZE", 8, int)
MAX_LENGTH = get_env_or_default("MAX_LENGTH", 512, int)
NUM_WORKERS = get_env_or_default("NUM_WORKERS", 2, int)
ENABLE_MODEL_CACHE = get_env_or_default("ENABLE_MODEL_CACHE", "true").lower() == "true"
MEMORY_THRESHOLD = get_env_or_default("MEMORY_THRESHOLD", 0.8, float)

# Topic modeling settings
MIN_TOPIC_SIZE = get_env_or_default("MIN_TOPIC_SIZE", 2, int)
NUM_TOPICS = get_env_or_default("NUM_TOPICS", 5, int)

# Visualization settings
FIGSIZE = (
    get_env_or_default("FIG_WIDTH", 10, int),
    get_env_or_default("FIG_HEIGHT", 6, int)
)
DPI = get_env_or_default("DPI", 100, int)
COLOR_PALETTE = get_env_or_default("COLOR_PALETTE", "viridis")

# Logging settings
LOG_LEVEL = get_env_or_default("LOG_LEVEL", "INFO")
LOG_FORMAT = get_env_or_default("LOG_FORMAT", "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
LOG_FILE = PROJECT_ROOT / "logs" / get_env_or_default("LOG_FILE", "car_reviews_analysis.log")
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

# Export settings
EXPORT_FORMATS = get_env_or_default("EXPORT_FORMATS", "json,csv,excel").split(",")
SAVE_PLOTS = get_env_or_default("SAVE_PLOTS", "true").lower() == "true"

# Model configurations dictionary for easy access
MODEL_CONFIGS = {
    "sentiment": {
        "model_name": SENTIMENT_MODEL,
        "task": "sentiment-analysis",
        "cache_key": f"sentiment_{SENTIMENT_MODEL}"
    },
    "translation": {
        "model_name": TRANSLATION_MODEL,
        "task": "translation",
        "cache_key": f"translation_{TRANSLATION_MODEL}"
    },
    "summarization": {
        "model_name": SUMMARIZATION_MODEL,
        "task": "summarization",
        "cache_key": f"summarization_{SUMMARIZATION_MODEL}"
    },
    "qa": {
        "model_name": QA_MODEL,
        "task": "question-answering",
        "cache_key": f"qa_{QA_MODEL}"
    },
    "aspect": {
        "model_name": ASPECT_SENTIMENT_MODEL,
        "task": "sentiment-analysis",
        "cache_key": f"aspect_{ASPECT_SENTIMENT_MODEL}"
    }
}

def get_model_config(task: str) -> Dict[str, str]:
    """Get model configuration for a specific task."""
    return MODEL_CONFIGS.get(task, {})

def get_all_configs() -> Dict[str, Any]:
    """Get all configuration settings as a dictionary."""
    return {
        "project_root": str(PROJECT_ROOT),
        "dataset_path": str(DATASET_PATH),
        "results_dir": str(RESULTS_DIR),
        "models": MODEL_CONFIGS,
        "performance": {
            "batch_size": BATCH_SIZE,
            "max_length": MAX_LENGTH,
            "num_workers": NUM_WORKERS,
            "enable_cache": ENABLE_MODEL_CACHE,
            "memory_threshold": MEMORY_THRESHOLD
        },
        "visualization": {
            "figsize": FIGSIZE,
            "dpi": DPI,
            "color_palette": COLOR_PALETTE
        },
        "export": {
            "formats": EXPORT_FORMATS,
            "save_plots": SAVE_PLOTS
        }
    }

# Environment validation
def validate_config():
    """Validate configuration settings."""
    errors = []
    
    if not DATASET_PATH.exists():
        errors.append(f"Dataset file not found: {DATASET_PATH}")
    
    if BATCH_SIZE <= 0:
        errors.append(f"Invalid batch size: {BATCH_SIZE}")
    
    if not 0 < MEMORY_THRESHOLD <= 1:
        errors.append(f"Invalid memory threshold: {MEMORY_THRESHOLD}")
    
    if errors:
        raise ValueError(f"Configuration errors: {'; '.join(errors)}")

# Auto-validate on import
try:
    validate_config()
except ValueError as e:
    print(f"Warning: {e}")  # Don't fail import, just warn
