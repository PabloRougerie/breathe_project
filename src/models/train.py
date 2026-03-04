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

    fit_time= round(fit_time,2)


    return model, fit_time
