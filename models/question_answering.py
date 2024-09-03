import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering

def question_answering(context, question, model_ckp="deepset/minilm-uncased-squad2"):
    tokenizer = AutoTokenizer.from_pretrained(model_ckp)
    model = AutoModelForQuestionAnswering.from_pretrained(model_ckp)
    
    inputs = tokenizer(question, context, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    
    start_idx = torch.argmax(outputs.start_logits)
    end_idx = torch.argmax(outputs.end_logits) + 1
    answer_span = inputs["input_ids"][0][start_idx:end_idx]
    return tokenizer.decode(answer_span)
