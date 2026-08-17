# -*- coding: utf-8 -*-
"""
Created on Sun Aug 16 12:30:26 2026

@author: michi
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from NN_simple import TimeSeriesData


X_train_raw = pd.read_csv("X_train.csv")[:]
y_train_raw = pd.read_csv("y_train.csv")[:]
y_train_raw["label"] = y_train_raw["target"] > 0

X_test = pd.read_csv("X_test.csv")

allocs = X_train_raw["GROUP"].unique()
























