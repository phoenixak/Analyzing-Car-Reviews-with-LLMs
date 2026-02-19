---
title: Car Reviews Analyzer
emoji: "\U0001F697"
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: "5.12.0"
app_file: app.py
pinned: false
---

# Car Reviews Analyzer

A comprehensive NLP analysis tool for automotive customer feedback. This Hugging Face Space demonstrates seven NLP capabilities applied to car review text.

## Capabilities

1. **Sentiment Analysis** -- Classify reviews as positive or negative with confidence scores
2. **Topic Modeling** -- Discover themes across multiple reviews using BERTopic
3. **Translation** -- Translate reviews from English to Spanish
4. **Question Answering** -- Ask factual questions about review text (extractive QA)
5. **Summarization** -- Generate concise summaries of lengthy reviews
6. **Named Entity Recognition** -- Extract people, organizations, and locations
7. **Aspect-Based Sentiment** -- Detect automotive aspects (performance, comfort, safety, etc.) and analyze sentiment per aspect

## Models Used

| Task | Model |
|------|-------|
| Sentiment Analysis | `distilbert-base-uncased-finetuned-sst-2-english` |
| Topic Modeling | BERTopic + `all-MiniLM-L6-v2` |
| Translation (EN to ES) | `Helsinki-NLP/opus-mt-en-es` |
| Question Answering | `distilbert-base-cased-distilled-squad` |
| Summarization | `t5-small` |
| Named Entity Recognition | `dbmdz/bert-large-cased-finetuned-conll03-english` |
| Aspect Analysis | Keyword detection + DistilBERT sentiment |

Models load on demand -- only the model needed for the selected tab is downloaded and cached on first use.

## Full Project

This Space is a demo component of the larger project:
[Analyzing Car Reviews with LLMs](https://github.com/Phoenixak99/Analyzing-Car-Reviews-with-LLMs)

## Run Locally

```bash
git clone https://huggingface.co/spaces/Phoenixak99/car-reviews-analyzer
cd car-reviews-analyzer
pip install -r requirements.txt
python app.py
```

The app will be available at `http://localhost:7860`.

## License

MIT
