# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 18:15:44 2026

@author: michi

This file uses Generalized Linear and Additive Models to predict whether
one should follow or short a certain asset allocation
"""

import sys
from pathlib import Path


# add the parent directory to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from evaluation import calculate_accuracy, confusion_matrix
from data.data_processing import get_data
from data.feature_engineering import featureEngineeringPipeline
import pandas as pd
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
import statsmodels.formula.api as smf

import os
import seaborn as sns
import matplotlib.pyplot as plt

# print(os.getcwd())

BASE_DIR = Path(__file__).resolve().parent.parent  # directory where your script lives



# X_train, X_test, y_train = get_data()




sns.set(style="whitegrid", palette="muted", context="notebook")



data = featureEngineeringPipeline(
    pd.read_csv(BASE_DIR / "data" / "X_train_cleaned.csv"), 
    pd.read_csv(BASE_DIR / "data" / "y_train_cleaned.csv"))


data.trim_cols()

features = data.features.copy()
response = data.response.copy()


# Model 1: simple GLM on naive features

features1 = pd.DataFrame({
    "RET_MEAN": features[data.ret_cols].mean(axis=1),
    "VOL_MEAN": features[data.vol_cols].mean(axis=1),
    "TURN_MEAN": features[data.turn_cols].mean(axis=1)
})

features1["ALLOCATION"] = features["ALLOCATION"]
features1["GROUP"] = features["GROUP"]

features1["TURN_MEAN"] = features1["TURN_MEAN"].fillna(features1["TURN_MEAN"].mean())


X_train, X_test, y_train, y_test = train_test_split(features1, response, test_size = 0.2, stratify = response["LABEL"])

train_data = X_train.join(y_train)

# sns.regplot(data=train_data, 
#             x="RET_MEAN", 
#             y="LABEL", 
#             y_jitter=0.01,
#             logistic=True,
#             ci = False)
# sns.boxplot(x="LABEL", y="RET_MEAN", data=train_data)
# sns.lmplot(
#     x="RET_MEAN",
#     y="TARGET",
#     hue="ALLOCATION",  # separate lines/colors for each category
#     data=train_data,
#     height=5,
#     aspect=1.2,
#     scatter_kws={"alpha":0.5}  # transparency for points
# )

sns.pairplot(train_data.drop("LABEL", axis=1),hue="ALLOCATION", diag_kind="kde")
plt.show()




gm1 = smf.glm("LABEL ~ RET_MEAN + VOL_MEAN + TURN_MEAN + C(GROUP)",
              data = train_data,
              family = sm.families.Binomial()
              ).fit()

print(gm1.summary())

yhat_train = gm1.predict()
yhat_train = (yhat_train > 0.5).astype(int)

print(f"train accuracy {calculate_accuracy(yhat_train, y_train['LABEL']):.4f}")

yhat_test = gm1.predict(X_test)
yhat_test = (yhat_test > 0.5).astype(int)

print(f"test accuracy {calculate_accuracy(yhat_test, y_test['LABEL']):.4f}")

print("confusion matrix")
print(confusion_matrix(yhat_test, y_test))





