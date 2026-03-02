from src.params import *
from src.utils import *
from src.preprocess.cleaning import *
from src.preprocess.features import *

import pandas as pd
import numpy as np
from sklearn.metrics import root_mean_squared_log_error, root_mean_squared_error


def baseline_calculation(X,y,baseline):

    X = X.sort_values(by= "date", ascending= True)

    if baseline not in ["persistence", "extrapolation", "average"]:
        raise ValueError(f"❌ baseline method must be either 'persistence', 'extrapolation', 'average'. {baseline} was passed instead")

    #persistence baseline
    if baseline == "persistence":
        y_pred = X["lag_1"]
        score = root_mean_squared_error(y, np.log1p(y_pred)) #y_true is log transformed already

    #extrapolation_baseline
    elif baseline == "extrapolation":
        y_pred = (X["lag_1"] + (X["lag_1"] - X["lag_2"])).clip(lower= 0.001)
        score = root_mean_squared_error(y, np.log1p(y_pred))

    #average baseline
    elif baseline == "average":
        y_pred = X[["lag_1", "lag_2", "lag_3"]].mean(axis= 1)
        score = root_mean_squared_error(y, np.log1p(y_pred))

    return score

def baseline_crossval(X,y,tscv):

    persistence_scores = []
    extrapolation_scores = []
    average_scores = []
    for train_idx, val_idx in tscv.split(X):
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
        persistence_scores.append(baseline_calculation(X_val, y_val, "persistence"))
        extrapolation_scores.append(baseline_calculation(X_val, y_val, "extrapolation"))
        average_scores.append(baseline_calculation(X_val, y_val, "average"))

    results = pd.DataFrame.from_dict({"persistence_baseline": np.mean(persistence_scores),
                            "Extrapolation_baseline": np.mean(extrapolation_scores),
                            "Average_baseline": np.mean(average_scores) }, orient= "index")
    print("✅ Baseline calculated and averaged over all folds")

    return results
