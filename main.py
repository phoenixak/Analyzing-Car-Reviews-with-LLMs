import pandas as pd
from src.pipelines import SentimentAnalysisPipeline, TranslationPipeline, SummarizationPipeline
from src.models import QuestionAnsweringModel
from src.utils import load_data, calculate_metrics, calculate_bleu_score

def main():
    # Load data
    reviews, real_labels = load_data(r"dataset\car_reviews.csv")

    # Sentiment Analysis
    sentiment_pipeline = SentimentAnalysisPipeline()
    predicted_labels = sentiment_pipeline(reviews)
    calculate_metrics(real_labels, predicted_labels)

    # Translation
    translation_pipeline = TranslationPipeline()
    first_review = reviews[0]
    translated_review = translation_pipeline(first_review)
    print(f"Model translation:\n{translated_review}")

    # Load reference translations
    with open(r"dataset\reference_translations.txt", 'r') as file:
        lines = file.readlines()
    references = [line.strip() for line in lines]
    print(f"Spanish translation references:\n{references}")

    # Calculate BLEU score
    bleu_score = calculate_bleu_score(translated_review, references)
    print(bleu_score)

    # Question Answering
    question_answering_model = QuestionAnsweringModel()
    context = reviews[1]
    question = "What did he like about the brand?"
    answer = question_answering_model(context, question)
    print("Answer: ", answer)

    # Summarization
    summarization_pipeline = SummarizationPipeline()
    text_to_summarize = reviews[-1]
    summarized_text = summarization_pipeline(text_to_summarize)
    print(f"Summarized text:\n{summarized_text}")

if __name__ == "__main__":
    main()