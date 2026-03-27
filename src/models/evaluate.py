from sklearn.metrics import root_mean_squared_error

from src.params import *


def evaluate(model, X_val, y_true):
    """Compute RMSE of model predictions on (X_val, y_true)."""
    y_pred = model.predict(X_val)
    score = root_mean_squared_error(y_true, y_pred)
    return score


def self_compare(score_ref, score_new, margin=DRIFT_THRESHOLD):
    """
    Compare current model on reference vs new batch (drift detection).
    score_ref: current model score on reference test set.
    score_new: current model score on the new batch of data.
    Returns True if drift detected (score_new >= score_ref * (1 + margin)).
    """
    if score_new >= score_ref * (1 + margin/100):

        return True
    return False


def cross_compare(score_old, score_new, margin=IMPROVEMENT_THRESHOLD):
    """
    Compare current model vs new model on the same batch.
    score_old: current model score on the new batch (same as score_new in self_compare).
    score_new: new model score on that same batch.
    Returns True if the new model improved (score_new <= score_old * (1 - margin)).
    """
    if score_new <= (1 - margin/100) * score_old:
        print("new model improved performance")
        return True
    return False
