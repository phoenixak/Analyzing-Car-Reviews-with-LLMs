from transformers import pipeline

def translate_review(review, model_name="Helsinki-NLP/opus-mt-en-es", max_length=27):
    translator = pipeline("translation", model=model_name)
    return translator(review, max_length=max_length)[0]['translation_text']
