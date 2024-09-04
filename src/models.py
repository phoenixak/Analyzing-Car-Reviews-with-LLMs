from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import torch

class QuestionAnsweringModel:
    def __init__(self):
        self.model_ckp = "deepset/minilm-uncased-squad2"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_ckp)
        self.model = AutoModelForQuestionAnswering.from_pretrained(self.model_ckp)

    def __call__(self, context, question):
        inputs = self.tokenizer(question, context, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs)
        start_idx = torch.argmax(outputs.start_logits)
        end_idx = torch.argmax(outputs.end_logits) + 1
        answer_span = inputs["input_ids"][0][start_idx:end_idx]
        return self.tokenizer.decode(answer_span)