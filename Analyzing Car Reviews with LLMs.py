!pip install transformers
!pip install evaluate
!pip install torch
from transformers import logging
logging.set_verbosity(logging.WARNING)
import pandas as pd
import torch

# Load the car reviews dataset
file_path = "D:\Machine Learning\Analyzing Car Reviews with LLMs\dataset\car_reviews.csv"
df = pd.read_csv(file_path, delimiter=";")

# Put the car reviews and their associated sentiment labels in two lists
reviews = df['Review'].tolist()
real_labels = df['Class'].tolist()



# Load a sentiment analysis LLM into a pipeline
from transformers import pipeline
classifier = pipeline('sentiment-analysis', model='distilbert-base-uncased-finetuned-sst-2-english')

# Perform inference on the car reviews and display prediction results
predicted_labels = classifier(reviews)
for review, prediction, label in zip(reviews, predicted_labels, real_labels):
    print(f"Review: {review}\nActual Sentiment: {label}\nPredicted Sentiment: {prediction['label']} (Confidence: {prediction['score']:.4f})\n")

# Load accuracy and F1 score metrics    
import evaluate
accuracy = evaluate.load("accuracy")
f1 = evaluate.load("f1")

# Map categorical sentiment labels into integer labels
references = [1 if label == "POSITIVE" else 0 for label in real_labels]
predictions = [1 if label['label'] == "POSITIVE" else 0 for label in predicted_labels]

# Calculate accuracy and F1 score
accuracy_result_dict = accuracy.compute(references=references, predictions=predictions)
accuracy_result = accuracy_result_dict['accuracy']
f1_result_dict = f1.compute(references=references, predictions=predictions)
f1_result = f1_result_dict['f1']
print(f"Accuracy: {accuracy_result}")
print(f"F1 result: {f1_result}")



# Load translation LLM into a pipeline and translate car review
first_review = reviews[0]
translator = pipeline("translation", model="Helsinki-NLP/opus-mt-en-es")
translated_review = translator(first_review, max_length=27)[0]['translation_text']
print(f"Model translation:\n{translated_review}")

# Load reference translations from file
with open("D:\Machine Learning\Analyzing Car Reviews with LLMs\dataset\reference_translations.txt", 'r') as file:
    lines = file.readlines()
references = [line.strip() for line in lines]
print(f"Spanish translation references:\n{references}")

# Load and calculate BLEU score metric
bleu = evaluate.load("bleu")
bleu_score = bleu.compute(predictions=[translated_review], references=[references])
print(bleu_score['bleu'])



# Import auto classes (optional: can be solved via pipelines too)
from transformers import AutoTokenizer
from transformers import AutoModelForQuestionAnswering

# Instantiate model and tokenizer
model_ckp = "deepset/minilm-uncased-squad2"
tokenizer = AutoTokenizer.from_pretrained(model_ckp)
model = AutoModelForQuestionAnswering.from_pretrained(model_ckp)

# Define context and question, and tokenize them
context = reviews[1]
print(f"Context:\n{context}")
question = "What did he like about the brand?"
inputs = tokenizer(question, context, return_tensors="pt")

# Perform inference and extract answer from raw outputs
with torch.no_grad():
  outputs = model(**inputs)
start_idx = torch.argmax(outputs.start_logits)
end_idx = torch.argmax(outputs.end_logits) + 1
answer_span = inputs["input_ids"][0][start_idx:end_idx]

# Decode and show answer
answer = tokenizer.decode(answer_span)
print("Answer: ", answer)



# Get original text to summarize upon car review
text_to_summarize = reviews[-1]
print(f"Original text:\n{text_to_summarize}")

# Load summarization pipeline and perform inference
model_name = "cnicu/t5-small-booksum"
summarizer = pipeline("summarization", model=model_name)
outputs = summarizer(text_to_summarize, max_length=53)
summarized_text = outputs[0]['summary_text']
print(f"Summarized text:\n{summarized_text}")
