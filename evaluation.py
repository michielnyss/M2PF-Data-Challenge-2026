# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 21:15:40 2026

@author: michi
"""
import numpy as np
import pandas as pd

def calculate_accuracy(pred, real):
    
    return np.sum(np.sign(pred)==np.sign(real)) / len(real)
    


def confusion_matrix(pred, real):
    
    FN = ((pred == 0) & (real == 1)).sum()
    FP = ((pred == 1) & (real == 0)).sum()
    TN = ((pred == 0) & (real == 0)).sum()
    TP = ((pred == 1) & (real == 1)).sum()
    
    conf_matrix = pd.DataFrame(
        [[TN, FP],
         [FN, TP]],
        index=["Actual 0", "Actual 1"],
        columns=["Predicted 0", "Predicted 1"]
        )
    
    return conf_matrix