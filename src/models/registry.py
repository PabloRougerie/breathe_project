
import mlflow
from mlflow.tracking import MlflowClient

from src.params import *


def load_model(mlflow_client, alias="champion"):
    """Load a registered model by alias. Returns (model, model_version)."""
    model_version = mlflow_client.get_model_version_by_alias(
        name=MLFLOW_MODEL_NAME, alias=alias
    ).version

    model = mlflow.sklearn.load_model(f"models:/{MLFLOW_MODEL_NAME}@{alias}")

    return model, model_version



def save_model(model):
    """Log model to MLflow and return its registered version number."""
    result = mlflow.sklearn.log_model(
        model=model,
        artifact_path="models",
        registered_model_name=MLFLOW_MODEL_NAME
    )
    return result.registered_model_version

def register_challenger(client, version):

     client.set_registered_model_alias(name= MLFLOW_MODEL_NAME,
                                      alias= "challenger",
                                      version= version)



def promote_challenger(client):

    # test if there is a champion already
    try:
        version_champ = client.get_model_version_by_alias(name= MLFLOW_MODEL_NAME, alias= "champion").version
        client.set_model_version_tag(name= MLFLOW_MODEL_NAME, version= version_champ, key= "status", value= "archived")
        print(f"   archived previous champion v{version_champ}")

    except mlflow.exceptions.MlflowException:
        print("   no existing champion — first promotion")

    version_chall = client.get_model_version_by_alias(name= MLFLOW_MODEL_NAME, alias= "challenger").version
    client.set_registered_model_alias(name= MLFLOW_MODEL_NAME,
                                      alias= "champion",
                                      version= version_chall)
    print(f"✅ Challenger v{version_chall} promoted to champion")
