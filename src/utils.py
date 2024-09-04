import pandas as pd
from evaluate import load

def load_data(file_path):
    df = pd.read_csv(file_path, delimiter=";")
    reviews = df['Review'].tolist()
    real_labels = df['Class'].tolist()
    return reviews, real_labels

def calculate_metrics(real_labels, predicted_labels):
    accuracy = load("accuracy")
    f1 = load("f1")
    references = [1 if label == "POSITIVE" else 0 for label in real_labels]
    predictions = [1 if label['label'] == "POSITIVE" else 0 for label in predicted_labels]
    accuracy_result = accuracy.compute(references=references, predictions=predictions)['accuracy']
    f1_result = f1.compute(references=references, predictions=predictions)['f1']
    print(f"Accuracy: {accuracy_result}")
    print(f"F1 result: {f1_result}")

def calculate_bleu_score(translated_review, references):
    bleu = load("bleu")
    return bleu.compute(predictions=[translated_review], references=[references])['bleu']