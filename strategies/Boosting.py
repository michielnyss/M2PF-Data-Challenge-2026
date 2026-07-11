# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 14:17:31 2026

@author: michi
"""

import sys
from pathlib import Path


# add the parent directory to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from data.data_processing import get_data
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, mean_squared_error
from category_encoders import TargetEncoder
from sklearn.model_selection import GroupShuffleSplit, GroupKFold, GridSearchCV



X_raw, _, y_raw = get_data()

test = (X_raw.copy()).columns

"""
----------------------
       Cleaning
----------------------
"""

num_cols = ([f"RET_{num}" for num in range(20, 0, -1)]
        + [f"SIGNED_VOLUME_{num}" for num in range(20, 0, -1)])

cols = ["ALLOCATION","GROUP","MEDIAN_DAILY_TURNOVER_1"] + num_cols

X_raw = X_raw[cols]
y = y_raw.copy()


# Feature engineering

def make_features(df):
    X = df[["ALLOCATION", "MEDIAN_DAILY_TURNOVER_1", "GROUP"]].copy()
    X = pd.get_dummies(X, columns=["GROUP"], drop_first=True)

    windows = [1, 3, 5, 10, 15, 20]
    for i in windows:
        ret_cols = [f"RET_{j}" for j in range(1, i + 1)]
        vol_cols = [f"SIGNED_VOLUME_{j}" for j in range(1, i + 1)]
        X[f"ret_mean_{i}"] = df[ret_cols].mean(axis=1)
        X[f"ret_std_{i}"] = df[ret_cols].std(axis=1, ddof=0)
        X[f"vol_mean_{i}"] = df[vol_cols].mean(axis=1)
        X[f"vol_std_{i}"] = df[vol_cols].std(axis=1, ddof=0)
    return X


"""
----------------------
       Training
----------------------
"""

# 1. Data Preprocessing

# 1a. Split data, keep allocation structure

gss = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
train_idx, test_idx = next(gss.split(X_raw, y, groups=X_raw["ALLOCATION"]))

X_raw_train = X_raw.iloc[train_idx].copy()
X_raw_test  = X_raw.iloc[test_idx].copy()
y_train     = y.iloc[train_idx].copy()
y_test      = y.iloc[test_idx].copy()

# 1b. Interpolate NaNs
X_raw_train[cols] = X_raw_train[cols].interpolate(method='linear', limit_direction='both')
X_raw_test[cols] = X_raw_test[cols].interpolate(method='linear', limit_direction='both')

# 1c. Create X features
X_train = make_features(X_raw_train)
X_test = make_features(X_raw_test)

# 1d. Align in case some GROUP categories are missing in one split
X_train, X_test = X_train.align(X_test, join='left', axis=1, fill_value=0)


# 1e. Save groups before target encoding overwrites ALLOCATION
train_groups = X_train["ALLOCATION"].values

# 1f. Target encoding of Allocation
encoder = TargetEncoder(cols="ALLOCATION")

X_train["ALLOCATION"] = encoder.fit_transform(X_train["ALLOCATION"], y_train["TARGET"])
X_test["ALLOCATION"]  = encoder.transform(X_test["ALLOCATION"])


# 2. Fitting

# 2a. Model 1: Simple Boosting
gbt = GradientBoostingClassifier(
    n_estimators=100,
    max_depth=2)

# gbt.fit(X_train, y_train["LABEL"])

# y_pred = gbt.predict(X_test)
# score  = accuracy_score(y_test["LABEL"], y_pred)
# print(f"Model 1 accuracy: {score:.4f}")


# # 2b. Model 2: Tuned with group-aware CV
# params_gbt = {
#     "n_estimators":  [100, 300, 500],
#     "max_depth":     [1, 2, 3],
#     "learning_rate": [0.05, 0.1, 0.15]
# }

# grid_gbt = GridSearchCV(estimator=GradientBoostingClassifier(),
#                         param_grid=params_gbt,
#                         cv=GroupKFold(n_splits=3),
#                         scoring="accuracy",
#                         verbose=1,
#                         n_jobs=-1)

# grid_gbt.fit(X_train, y_train["LABEL"], groups=train_groups)

# best_hyper_param = grid_gbt.best_params_
# print("Best hyper parameters:", best_hyper_param)
# print(f"Best CV accuracy: {grid_gbt.best_score_:.4f}")


best_hyper = {'learning_rate': 0.1, 'max_depth': 3, 'n_estimators': 500}

gbt = GradientBoostingClassifier(**best_hyper)

gbt.fit(X_train, y_train["LABEL"])

y_pred = gbt.predict(X_test)
score  = accuracy_score(y_test["LABEL"], y_pred)
print(f"Model 1 accuracy: {score:.4f}")

importance = gbt.feature_importances_
indices = np.argsort(importance)

plt.barh(range(len(indices)), importance[indices])
plt.yticks(range(len(indices)), X_train.columns[indices])
plt.xlabel("Feature Importance")
plt.title("Feature Importance (Gradient Boosting)")
plt.show()






