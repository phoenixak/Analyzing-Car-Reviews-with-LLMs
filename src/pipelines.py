from transformers import pipeline

class SentimentAnalysisPipeline:
    def __init__(self):
        self.classifier = pipeline('sentiment-analysis', model='distilbert-base-uncased-finetuned-sst-2-english')

    def __call__(self, reviews):
        return self.classifier(reviews)

class TranslationPipeline:
    def __init__(self):
        self.translator = pipeline("translation", model="Helsinki-NLP/opus-mt-en-es")

    def __call__(self, text):
        return self.translator(text, max_length=400)[0]['translation_text']


class SummarizationPipeline:
    def __init__(self):
        self.summarizer = pipeline("summarization", model="cnicu/t5-small-booksum")

    def __call__(self, text):
        return self.summarizer(text, max_length=53)[0]['summary_text']