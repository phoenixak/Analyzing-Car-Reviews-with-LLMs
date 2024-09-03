from data.data_loader import load_reviews, load_reference_translations
from models.sentiment_analysis import sentiment_analysis
from models.translation import translate_review
from models.question_answering import question_answering
from models.summarization import summarize_text
from models.evaluate_metrics import compute_metrics
from utils.logger import setup_logger
from utils.mlflow_tracking import log_model_to_mlflow

def main():
    logger = setup_logger()

    # Load data
    file_path = "dataset/car_reviews.csv"
    translations_path = "dataset/reference_translations.txt"
    logger.info("Loading data...")
    reviews = load_reviews(file_path)['Review'].tolist()
    real_labels = load_reviews(file_path)['Class'].tolist()
    references = load_reference_translations(translations_path)

    # Sentiment Analysis
    logger.info("Performing sentiment analysis...")
    predicted_labels = sentiment_analysis(reviews)
    predictions = [1 if label['label'] == "POSITIVE" else 0 for label in predicted_labels]
    references = [1 if label == "POSITIVE" else 0 for label in real_labels]
    accuracy_result = compute_metrics(references, predictions, "accuracy")
    f1_result = compute_metrics(references, predictions, "f1")
    logger.info(f"Accuracy: {accuracy_result['accuracy']}, F1: {f1_result['f1']}")
    
    # Use a list containing a single string as the input example
    input_example = [reviews[0]]

    # Log the sentiment analysis model to MLflow
    log_model_to_mlflow(
        model_name="sentiment_analysis", 
        model=sentiment_analysis, 
        input_example=input_example,  # Pass a list with one string
        metrics={"accuracy": accuracy_result['accuracy'], "f1": f1_result['f1']}
    )

    # Translation
    logger.info("Performing translation...")
    translated_review = translate_review(reviews[0])
    logger.info(f"Translated review: {translated_review}")

    # Ensure references and predictions are in the correct format
    if isinstance(references[0], list):
        # Multiple references
        bleu_score = compute_metrics([translated_review], [references])
    else:
        # Single reference
        bleu_score = compute_metrics([translated_review], [[ref] for ref in references])

    logger.info(f"BLEU Score: {bleu_score['bleu']}")

    # Question Answering
    logger.info("Performing question answering...")
    context = reviews[1]
    question = "What did he like about the brand?"
    answer = question_answering(context, question)
    logger.info(f"Answer: {answer}")

    # Summarization
    logger.info("Performing summarization...")
    summarized_text = summarize_text(reviews[-1])
    logger.info(f"Summarized text: {summarized_text}")

if __name__ == "__main__":
    main()
