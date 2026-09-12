import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix


def gain_score(y: pd.Series, y_pred: pd.Series, **kwargs) -> np.float64:
    weight_fp = kwargs.get("weight_fp", 10)
    weight_fn = kwargs.get("weight_fn", 1)

    cm = confusion_matrix(y, y_pred)
    gm = np.array([[0, -weight_fp], [-weight_fn, 0]])

    return np.sum(cm * gm)
