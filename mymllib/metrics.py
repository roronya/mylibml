import numpy as np
from sklearn.metrics import mean_squared_error

def dcg(true, pred, K=None):
    pred_arg_sorted = pred.argsort()[::-1]
    true_sorted = true[pred_arg_sorted]
    N = true.shape[0]
    K = N if K is None else K
    return np.sum([true_sorted[i]/np.log2(i+1) if i != 0 else true_sorted[i] for i in range(K)])

def ndcg(true, pred, K=None):
    return dcg(true, pred, K=K)/dcg(true, true, K=K)

def propensity_scored_mse(y_true, y_pred, propensity_score):
    return np.average(1/propensity_score * (y_true - y_pred)**2)
