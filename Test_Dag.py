import mlflow

# Set the tracking URI to the DagsHub repository
mlflow.set_tracking_uri("https://dagshub.com/Jatin-Repository/MLflow_Learning.mlflow")

# Start an MLflow run
with mlflow.start_run() as run:
    # Log parameters, metrics, etc.
    mlflow.log_param("param_name", param_value)
    mlflow.log_metric("metric_name", metric_value)

# Get the run ID
run_id = run.info.run_id
print(f"Run ID: {run_id}")
