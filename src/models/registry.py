
import mlflow
from mlflow.tracking import MlflowClient
from datetime import datetime
from src.params import *


def load_model(mlflow_client, alias= "champion"):

    # get model metadata

    version = mlflow_client.get_model_version_by_alias(name= MLFLOW_MODEL_NAME, alias= alias)
    time = datetime.fromtimestamp(version.creation_timestamp / 1000) #conv ms into s
    model_metadata = {"model_version": version.version,
                      "model_run_id": version.run_id,
                      "model_date": time}

    model = mlflow.sklearn.load_model(f"models:/{MLFLOW_MODEL_NAME}@{alias}")


    return model, model_metadata



def save_model(model):

    #save model to mlflow bucket and return model info
    result = mlflow.sklearn.log_model(
        model= model,
        artifact_path="models",
        registered_model_name=MLFLOW_MODEL_NAME
    )

    #get model metadata
    model_metadata = {"model_version": result.registered_model_version,
                      "model_run_id": result.run_id,
                      "model_log_date": result.utc_time_created}

    return model_metadata

def register_challenger(client, version):

     client.set_registered_model_alias(name= MLFLOW_MODEL_NAME,
                                      alias= "challenger",
                                      version= version)



def promote_challenger(client):

    # test if there is a champion already
    try:
        version_champ = client.get_model_version_by_alias(name= MLFLOW_MODEL_NAME, alias= "champion").version
        # if exists, tag it to "archived"
        client.set_model_version_tag(name= MLFLOW_MODEL_NAME, version= version_champ, key= "status", value= "archived")

    except mlflow.exceptions.MlflowException: #if mlflow throws exception
        pass #if (first promotion) this part is skipped.

    version_chall = client.get_model_version_by_alias(name= MLFLOW_MODEL_NAME, alias= "challenger").version
    client.set_registered_model_alias(name= MLFLOW_MODEL_NAME,
                                      alias= "champion",
                                      version= version_chall)
