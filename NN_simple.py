# -*- coding: utf-8 -*-
"""
Created on Mon Aug  3 12:22:31 2026

@author: michi
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.nn.functional as F


class TimeSeriesData(Dataset):
    
    def __init__(self, X, y):
        X = X.interpolate(axis=1).bfill(axis=1).to_numpy() # bfill: leading NaNs

        X_mean, X_std = X.mean(axis=1), X.std(axis=1) # standardize row wise

        self.X = torch.tensor(
            (X - X_mean[:, None]) / X_std[:, None],
            dtype=torch.float32
            )

        self.y = torch.tensor(
            y.to_numpy(),
            dtype=torch.float32
            ).unsqueeze(1) # (N, 1) to match the model output

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    
    
class simpleNet(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(20, 10)
        self.fc2 = nn.Linear(10, 5)
        self.fc3 = nn.Linear(5, 1)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.sigmoid(self.fc3(x))
        return x


def calc_accuracy(prediction, target):
    correct = ((prediction > 0.5).float() == target).sum().item()
    total = target.numel()
    return correct/total


def train(model, optimizer, loss_fn, train_loader, val_loader,
          epochs=1):
    
    train_loss_lst = []
    train_acc_lst = []
    val_loss_lst = []
    val_acc_lst = []
    
    for epoch in range(epochs):
        train_loss = 0
        train_acc = 0
        model.train()
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            prediction = model(X_batch)
            loss = loss_fn(prediction, y_batch)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.data.item()
            train_acc += calc_accuracy(prediction, y_batch)
            
        train_loss_lst.append(train_loss/len(train_loader))
        train_acc_lst.append(train_acc/len(train_loader))
        
        
        val_loss = 0
        val_acc = 0
        model.eval()
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                prediction = model(X_batch)
                loss = loss_fn(prediction, y_batch)

                val_loss += loss.data.item()
                val_acc += calc_accuracy(prediction, y_batch)

        val_loss_lst.append(val_loss/len(val_loader))
        val_acc_lst.append(val_acc/len(val_loader))

        print(f"Epoch {epoch+1} | Loss: {train_loss_lst[-1]:.4f} | Accuracy: {train_acc_lst[-1]:.2f}")
        
    losses = pd.DataFrame({
        "train_loss": train_loss_lst,
        "train_acc": train_acc_lst,
        "val_loss": val_loss_lst,
        "val_acc": val_acc_lst
        })
    
    return model, losses

def load_model():
    
    simplenet = simpleNet()
    simplenet_state_dict = torch.load("simplenet.pt")
    simplenet.load_state_dict(simplenet_state_dict) # loads in place

    return simplenet

# X_train_raw = pd.read_csv("X_train.csv")
# y_train_raw = pd.read_csv("y_train.csv")
# y_train_raw["label"] = y_train_raw["target"] > 0

# X_test = pd.read_csv("X_test.csv")

# RET_COLS = [f"RET_{i+1}" for i in range(19, -1, -1)]
# VOL_COLS = [f"SIGNED_VOLUME_{i+1}" for i in range(19, -1, -1)]

# # data prep
# batch_size = 228

# X_train, X_val, y_train, y_val = train_test_split(X_train_raw[RET_COLS], y_train_raw["label"])

# train_dataset = TimeSeriesData(X_train, y_train)
# train_data_loader = DataLoader(train_dataset, batch_size)
# val_dataset = TimeSeriesData(X_val, y_val)
# val_data_loader = DataLoader(val_dataset, batch_size)

# # training
# model = simpleNet()
# optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
# model, losses = train(model, optimizer, nn.BCELoss(),
#                       train_data_loader, val_data_loader, 10)

# torch.save(model.state_dict(), "simplenet.pt")

model = load_model()

# for col in losses.columns:
#     plt.plot(losses[col], label=col)
    
plt.legend()
plt.show()









