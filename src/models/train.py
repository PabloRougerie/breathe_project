from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer
from lightgbm import LGBMRegressor
from src.params import *
import time

def initiate_model():
    encoder  = make_column_transformer((OneHotEncoder(sparse_output= False, handle_unknown= "ignore"),
                                   ["city"]),
                                   remainder= "passthrough",
                                   verbose_feature_names_out= False)
    pipe = make_pipeline(encoder, LGBMRegressor(**BEST_PARAMS))

    return pipe


def train_model(model, X, y):

    t0 = time.time()
    model.fit(X,y)
    fit_time = time.time() - t0

    metadata= {
        "n_rows": len(X),
        "n_features": len(X.columns),
        "features_name": X.columns.tolist(),
        "fit_time_seconds": round(fit_time,2)
    }

    return model, metadata



    # 2. .fit() on X, y
    # 3. return fitted model
    # Optional: create a dict of metadata to be logged later on when used with mlflow decorator?
