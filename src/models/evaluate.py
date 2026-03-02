
from sklearn.metrics import root_mean_squared_error
from src.params import *

def evaluate(model, X_val, y_true):

    y_pred = model.predict(X_val)
    score = root_mean_squared_error(y_true, y_pred)
    return score

def self_compare(score_ref, score_test, margin= DRIFT_THRESHOLD):

    if score_test >= score_ref*(1+ margin):
        print(f"drift detected...")

        is_drift = True
        return is_drift
    else:
        is_drift = False
        return is_drift




def cross_compare(score_old, score_new, margin= IMPROVEMENT_THRESHOLD):

    if score_new <= (1- margin) * score_old:
        print(f"new model improved performance")
        is_better = True
        return is_better

    else:
        is_better= False
        return is_better
