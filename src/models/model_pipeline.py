import mlflow
from datetime import datetime
from src.params import *
from src.models.train import *
from src.models.evaluate import *
from src.models.registry import *


def setup_mlflow():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    experiment = mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    print(f"✅ MLflow ready")
    print(f"   tracking URI : {mlflow.get_tracking_uri()}")
    print(f"   experiment   : {experiment.name} (id: {experiment.experiment_id})")



def run_training(X, y, dataset_metadata):
    """Instantiate, train, save and register a new model as 'challenger'.

    Logs to MLflow:
      params   — hyperparams (BEST_PARAMS), model_version, date_start, date_end
      artifact — train_metadata.json: fit_time_seconds, n_rows, n_features, list_features
    Returns:
      trained_model, model_version (str)
    """
    mlflow.end_run()

    with mlflow.start_run() as run:

        client = MlflowClient()
        print(f"ℹ️ MLflow run started — run_id: {run.info.run_id}")

        print("📦 Instantiating model...")
        model = initiate_model()

        print(f"⚙️  Fitting on {len(X)} rows, {len(X.columns)} features...")
        trained_model, fit_time = train_model(model, X, y)
        print(f"   fit time: {fit_time}s")

        # Filterable params
        mlflow.log_params(BEST_PARAMS)

        # Single artifact: fit context + dataset stats
        train_artifact = {
            "fit_time": fit_time,
            **{k: v for k, v in dataset_metadata.items() if k not in ("date_start", "date_end")}
            # n_rows, n_features, list_features
        }
        mlflow.log_dict(train_artifact, "train_metadata.json")

        print("💾 Saving model to MLflow registry...")
        model_version = save_model(trained_model)
        print(f"   model registered as version {model_version}")

        mlflow.log_params({
            "date_start":    dataset_metadata["date_start"],
            "date_end":      dataset_metadata["date_end"],
            "model_version": model_version
        })


        try:
            client.get_model_version_by_alias(name= MLFLOW_MODEL_NAME, alias= "champion")
            print(f"no champion found. registring model as first champion")
            #if works, there's a champion, registring a challenger
            assigned_alias = "challenger"
            register_model(client, version=model_version, alias= assigned_alias)
            mlflow.set_tags({
            "run_type":    "training",
            "model_alias": assigned_alias,
            "date_start":  dataset_metadata["date_start"],
            "date_end":    dataset_metadata["date_end"],
        })
            print(f"✅ Model v{model_version} trained, logged and registered as {assigned_alias}")

        except:
            #if doesn't work: no champ yet (first train), promoting as champion
            assigned_alias = "champion"
            register_model(client, version=model_version, alias= assigned_alias)
            mlflow.set_tags({
            "run_type":    "training",
            "model_alias": assigned_alias,
            "date_start":  dataset_metadata["date_start"],
            "date_end":    dataset_metadata["date_end"],
        })
            print(f"✅ Model v{model_version} trained, logged and registered as '{assigned_alias}'")




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



    with mlflow.start_run() as run:
        print(f"ℹ️ MLflow run started — run_id: {run.info.run_id}")

        # Load from registry if no model passed (standalone eval, not chained after training)
        if model is None:
            print(f"📦 Loading '{alias}' model from registry...")
            client = MlflowClient()
            model, model_version = load_model(client, alias=alias)
            print(f"   loaded model v{model_version}")

        print(f"⚙️ Evaluating on {len(X_val)} rows ({eval_mode})...")
        score = evaluate(model, X_val, y_true)
        print(f"   RMSE: {round(score, 4)}")

        # All filterable params in one call
        mlflow.log_params({
            "eval_date":     str(datetime.now().date()),
            "eval_mode":     eval_mode,
            "model_version": model_version,
            "model_alias":   alias,
            "date_start":    dataset_metadata["date_start"],
            "date_end":      dataset_metadata["date_end"],
        })
        mlflow.log_metric("rmse", score)

        mlflow.set_tags({
            "run_type":    "evaluation",
            "model_alias": alias,
            "date_start":  dataset_metadata["date_start"],
            "date_end":    dataset_metadata["date_end"],
        })

        print(f"✅ Evaluation done — RMSE: {round(score, 4)} | model v{model_version} | {eval_mode}")

    return score
