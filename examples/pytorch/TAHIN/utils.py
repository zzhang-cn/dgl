import numpy as np
import torch
from sklearn.metrics import roc_auc_score, accuracy_score, log_loss, f1_score, average_precision_score, ndcg_score

def evaluate_auc(pred, label):
    res=roc_auc_score(y_score=pred, y_true=label)
    return res

def evaluate_acc(pred, label):
    res = []
    for _value in pred:
        if _value >= 0.5:
            res.append(1)
        else:
            res.append(0)
    return accuracy_score(y_pred=res, y_true=label)

def evaluate_f1_score(pred, label):
    res = []
    for _value in pred:
        if _value >= 0.5:
            res.append(1)
        else:
            res.append(0)
    return f1_score(y_pred=res, y_true=label)

def evaluate_logloss(pred, label):
    res = log_loss(y_true=label, y_pred=pred,eps=1e-7, normalize=True)
    return res
    