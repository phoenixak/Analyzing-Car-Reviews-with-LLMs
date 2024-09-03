from transformers import pipeline

def summarize_text(text, model_name="cnicu/t5-small-booksum", max_length=53):
    summarizer = pipeline("summarization", model=model_name)
    outputs = summarizer(text, max_length=max_length)
    return outputs[0]['summary_text']
