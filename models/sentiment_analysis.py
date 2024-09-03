from transformers import pipeline

def sentiment_analysis(reviews):
    classifier = pipeline('sentiment-analysis', model='distilbert-base-uncased-finetuned-sst-2-english')
    return classifier(reviews)
