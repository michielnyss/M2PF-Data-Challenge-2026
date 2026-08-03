# -*- coding: utf-8 -*-
"""
Created on Mon Aug  3 12:22:31 2026

@author: michi
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn

X_train = pd.read_csv("X_train.csv")
y_train = pd.read_csv("y_train.csv")
y_train["label"] = y_train["target"] > 0
X_test = pd.read_csv("X_test.csv")

RET_COLS = [f"RET_{i+1}" for i in range(19, -1, -1)]
VOL_COLS = [f"SIGNED_VOLUME_{i+1}" for i in range(19, -1, -1)]


class TimeSeriesDataSet(Dataset):
    
    def __init__(self, X, y):
        self.X = X.to_numpy(dtype=np.float32)
        self.y = y.to_numpy(dtype=np.float32)
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, i):
        return self.X[i], self.y[i]
    
    
class SimpleNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(20, 10),
            nn.ReLU(),
            nn.Linear(10,1),
            nn.Sigmoid()
            )
        
    def forward(self, x):
        return self.network(x)


def train(data):
    
    loader = DataLoader(
        data,
        batch_size=20,
        shuffle=True
        )
    
    model = SimpleNetwork()
    
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.01
        )

    losses = []
    accuracy = []
    for epoch in range(1):
        
        for X_batch, y_batch in loader:
        
            y_pred = model(X_batch).squeeze(-1)
            
            loss = nn.BCELoss()(
                y_pred,
                y_batch
                )
            losses.append(loss.item())
            
            correct = ((y_pred > 0.5).float() == y_batch).sum().item()
            accuracy.append(correct/y_batch.numel())
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
    
    return model, losses, accuracy

def evaluate(model, data):
    model.eval()
    loader = DataLoader(data, batch_size=20, shuffle=True)
    correct = total = 0
    
    for X_batch, y_batch in loader:
        preds = (model(X_batch).squeeze(-1) > 0.5).float()
        correct += (preds == y_batch).sum().item()
        total += y_batch.numel()
    model.train()
    return correct/total

data =  TimeSeriesDataSet(
    X = X_train.loc[:99, RET_COLS], 
    y = y_train.loc[:99, "label"]
    )

trained_model, losses, accuracy = train(data)
    

plt.plot(losses, label="BEC loss")
plt.plot(accuracy, label="accuracy")
plt.legend()
plt.show()





