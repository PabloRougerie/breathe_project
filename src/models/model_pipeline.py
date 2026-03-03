import mlflow
from src.params import *
from src.models.train import *
from src.models.evaluate import *
from src.models.registry import *


def setup_mlflow():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

def run_training(X,y):

    """ instantiate and train and save new model"""

    # close all existing runs
    mlflow.end_run()
    #open new run
    with mlflow.start_run():

        client = MlflowClient()
        #instantiate model
        print("📦 Instantiating new model...")
        model = initiate_model()

        #train model
        print(f"⚙️ Fitting new model on {len(X)} lines...")
        trained_model, train_metadata = train_model(model, X, y)

        #log params and model/dataset metadata
        mlflow.log_params(BEST_PARAMS)
        mlflow.log_dict(train_metadata,"train_metadata.json")


        #save model and set it to "challenger" alias
        print("saving...")
        model_metadata = save_model(trained_model)
        print(f"Registring new model v{model_metadata["model_version"]} as 'challenger'...")
        register_challenger(client, version= model_metadata["model_version"])
        print(f"✅ Model v{model_metadata["model_version"]} trained and registered")

        return trained_model, model_metadata





def run_evaluating(X_val, y_true, model= None, model_metadata= None, alias= "champion", data_source= "test_set"):
    """ load and evaluate existing model"""

    if data_source not in ("test_set", "fresh_batch"):
        raise ValueError(f"❌ data_source must be 'test_set' or 'fresh_batch', got {data_source} instead")

    #close existing run
    with mlflow.start_run():

        # check if model passed as argument (when chained with model train for instance)
        if model is None:
            print(f"no model passed. Loading {alias} model from registry")
            client = MlflowClient()
            # load model
            model, model_metadata = load_model(client, alias= alias)

        #eval model
        score = evaluate(model, X_val, y_true)
        if data_source =="test_set":
            mlflow.log_metric("test_rmse", score)
        else:
            mlflow.log_metric("fresh_batch_rmse", score)


    return score
