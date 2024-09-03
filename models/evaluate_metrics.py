import evaluate

def compute_metrics(references, predictions, metric_name="accuracy"):
    metric = evaluate.load(metric_name)
    return metric.compute(references=references, predictions=predictions)
