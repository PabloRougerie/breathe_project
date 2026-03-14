
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
        sk_model=model,
        artifact_path="models",
        registered_model_name=MLFLOW_MODEL_NAME
    )
    return result.registered_model_version

def register_model(client, version, alias: str):
    """Assign an alias ('champion' or 'challenger') to a registered model version and set matching tag."""

    if alias not in ["challenger", "champion"]:
        raise ValueError(f"alias must be 'challenger' or 'champion', got {alias} instead")


    client.set_registered_model_alias(name= MLFLOW_MODEL_NAME,
                                      alias= alias,
                                      version= version)
    client.set_model_version_tag(name= MLFLOW_MODEL_NAME, version= version, key= "alias", value= alias)



def promote_challenger():
    """Promote the current 'challenger' to 'champion'.

    Tags the existing champion as 'archived' and removes its alias before promoting.
    Safe on first promotion (no existing champion).
    """
    client = MlflowClient()

    try:
        # if works: there's a champ: need to archive it
        version_champ = client.get_model_version_by_alias(name= MLFLOW_MODEL_NAME, alias= "champion").version

        #tag it as "archived"
        client.set_model_version_tag(name= MLFLOW_MODEL_NAME, version= version_champ, key= "status", value= "archived")

        # remove its "alias" version tag (MLflow alias is reassigned automatically)
        client.delete_model_version_tag(name=MLFLOW_MODEL_NAME, version=version_champ, key="alias")
        print(f"   archived previous champion v{version_champ}")

    except mlflow.exceptions.MlflowException:
        print("   no existing champion: first promotion")

    # get version of the current challenger
    version_chall = client.get_model_version_by_alias(name= MLFLOW_MODEL_NAME, alias= "challenger").version
    # set its alias as champion.
    client.set_registered_model_alias(name= MLFLOW_MODEL_NAME,
                                      alias= "champion",
                                      version= version_chall)

    # remove the alias "challenger"
    client.delete_registered_model_alias(name=MLFLOW_MODEL_NAME, alias="challenger")

    # overwrite the "alias" version tag from "challenger" to "champion"
    client.set_model_version_tag(name= MLFLOW_MODEL_NAME, version= version_chall, key= "alias", value= "champion")
    print(f"✅ Challenger v{version_chall} promoted to champion")
