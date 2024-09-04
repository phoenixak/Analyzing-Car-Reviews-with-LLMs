import mlflow
import mlflow.sklearn
from src.pipelines import SentimentAnalysisPipeline, TranslationPipeline, SummarizationPipeline
from src.models import QuestionAnsweringModel
from src.utils import load_data, calculate_metrics, calculate_bleu_score

def main():
    # Initialize MLflow
    mlflow.start_run()

    # Load data
    reviews, real_labels = load_data(r"dataset\car_reviews.csv")

    # Sentiment Analysis
    sentiment_pipeline = SentimentAnalysisPipeline()
    predicted_labels = sentiment_pipeline(reviews)
    accuracy_result, f1_result = calculate_metrics(real_labels, predicted_labels)
    
    # Log metrics
    mlflow.log_metric("accuracy", accuracy_result)
    mlflow.log_metric("f1_score", f1_result)

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

    # Calculate and log BLEU score
    bleu_score = calculate_bleu_score(translated_review, references)
    mlflow.log_metric("bleu_score", bleu_score)
    print(f"BLEU score: {bleu_score}")

    # Question Answering
    question_answering_model = QuestionAnsweringModel()
    context = reviews[1]
    question = "What did he like about the brand?"
    answer = question_answering_model(context, question)
    print("Answer: ", answer)
    
    # Log model
    mlflow.log_artifact("path_to_your_model_artifact")  # Replace with the actual path to your model artifact

    # Summarization
    summarization_pipeline = SummarizationPipeline()
    text_to_summarize = reviews[-1]
    summarized_text = summarization_pipeline(text_to_summarize)
    print(f"Summarized text:\n{summarized_text}")

    # End the MLflow run
    mlflow.end_run()

if __name__ == "__main__":
    main()
