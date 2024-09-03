import mlflow
import mlflow.pyfunc

def log_model_to_mlflow(model_name, model, input_example, metrics):
    with mlflow.start_run(run_name=model_name):
        # Log the model with an input example
        mlflow.pyfunc.log_model(
            artifact_path=model_name,
            python_model=model,
            input_example=input_example,  # Provide an example input
            pip_requirements=["transformers", "torch"]  # Add any additional requirements if needed
        )
        mlflow.log_metrics(metrics)
