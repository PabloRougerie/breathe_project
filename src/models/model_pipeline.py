import mlflow
from datetime import datetime
from src.params import *
from src.models.train import *
from src.models.evaluate import *
from src.models.registry import *


def setup_mlflow():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)


def run_training(X, y, dataset_metadata):
    """Instantiate, train, save and register a new model as 'challenger'.

    Logs to MLflow:
      params   — hyperparams (BEST_PARAMS), model_version, date_start, date_end
      artifact — train_metadata.json: fit_time_seconds, n_rows, n_features, list_features
    Returns:
      trained_model, model_version (str)
    """
    mlflow.end_run()
    with mlflow.start_run():

        client = MlflowClient()

        print("📦 Instantiating new model...")
        model = initiate_model()

        print(f"⚙️ Fitting new model on {len(X)} lines...")
        trained_model, fit_time = train_model(model, X, y)  # train_metadata: fit_time_seconds

        # Filterable params
        mlflow.log_params(BEST_PARAMS)


        # Single artifact: fit context + dataset stats
        train_artifact = {
            "fit_time": fit_time,  # fit_time_seconds
            **{k: v for k, v in dataset_metadata.items() if k not in ("date_start", "date_end")}
            # n_rows, n_features, list_features
        }
        mlflow.log_dict(train_artifact, "train_metadata.json")

        #save model and set it to "challenger" alias
        print("saving...")
        model_version = save_model(trained_model)

        mlflow.log_params({
            "date_start": dataset_metadata["date_start"],  # actual start date after preprocessing
            "date_end":   dataset_metadata["date_end"],# actual end date after preprocessing
            "model_version": model_version
        })


        print(f"Registering new model v{model_version} as 'challenger'...")
        register_challenger(client, version=model_version)
        print(f"✅ Model v{model_version} trained and registered")

        return trained_model, model_version





def run_evaluating(X_val, y_true, dataset_metadata, model=None, model_version=None, alias="champion", eval_mode="test_set"):
    """Load and evaluate an existing model.

    Logs to MLflow:
      params  — eval_date, eval_mode, model_version, model_alias, date_start, date_end
      metric  — test_rmse (eval_mode='test_set') or fresh_batch_rmse (eval_mode='fresh_batch')
    Args:
      model, model_version: pass when chaining with run_training (avoids registry round-trip)
      alias: used when loading from registry (ignored if model is passed)
      eval_mode: 'test_set' (periodic benchmark) or 'fresh_batch' (new production data)
    Returns:
      score (float)
    """
    if eval_mode not in ("test_set", "fresh_batch"):
        raise ValueError(f"❌ eval_mode must be 'test_set' or 'fresh_batch', got '{eval_mode}'")

    with mlflow.start_run():

        # Load from registry if no model passed (standalone eval, not chained after training)
        if model is None:
            print(f"No model passed. Loading '{alias}' model from registry...")
            client = MlflowClient()
            model, model_version = load_model(client, alias=alias)

        score = evaluate(model, X_val, y_true)

        # All filterable params in one call
        mlflow.log_params({
            "eval_date":     str(datetime.now().date()),
            "eval_mode":     eval_mode,
            "model_version": model_version,
            "model_alias":   alias,
            "date_start":    dataset_metadata["date_start"],  # actual start date after preprocessing
            "date_end":      dataset_metadata["date_end"],    # actual end date after preprocessing
        })

        metric_name = "rmse"
        mlflow.log_metric(metric_name, score)

    return score
