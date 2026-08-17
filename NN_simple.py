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
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F

sns.set_theme()


class TimeSeriesData(Dataset):
    
    def __init__(self, X_RET, X_VOL, X_TURN, X_ALLOC, X_GROUP, y):
        
        def standardize_cont(X, axis=1):
            X = X.interpolate(axis=axis).bfill(axis=axis).to_numpy()
        
            X_mean = X.mean(axis=axis)
            X_std = X.std(axis=axis)
        
            if axis == 0:
                X = (X - X_mean[None, :]) / X_std[None, :]
            elif axis == 1:
                X = (X - X_mean[:, None]) / X_std[:, None]
            else:
                raise ValueError("axis must be 0 or 1")
        
            return torch.tensor(X, dtype=torch.float32)
        
        self.X_RET = standardize_cont(X_RET)
        self.X_VOL = standardize_cont(X_VOL)
        self.X_TURN = standardize_cont(X_TURN, axis=0)
        
        self.X_ALLOC = torch.tensor(X_ALLOC.to_numpy(), dtype=torch.long)
        self.X_GROUP = torch.tensor(X_GROUP.to_numpy(), dtype=torch.long)

        self.y = torch.tensor(
            y.to_numpy(),
            dtype=torch.float32
            ).unsqueeze(1) # (N, 1) to match the model output

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return (
            self.X_RET[idx],
            self.X_VOL[idx],
            self.X_TURN[idx],
            self.X_ALLOC[idx],
            self.X_GROUP[idx],
            self.y[idx]
            )
    
    
class evaluateModel():
    def __init__(self, model, losses):
        self.model = model
        self.losses = losses
        
    def print_loss(self, key = None):
        if key:
            print(self.losses[key])
        else:
            print(self.losses)
        
    def plot_loss(self, key = None):
        if key:
            cols = key
        else:
            cols = self.losses.columns
            
        for col in cols:
            plt.plot(self.losses[col], label=col)
        
        plt.legend()
        plt.show()
              
    
class simpleNet(nn.Module):
    
    def __init__(self):
        super().__init__()
        
        # Categorical
        
        self.alloc_emb = nn.Embedding(278, 8)
        
        self.group_emb = nn.Embedding(4, 4)
        
        # Continuous
        
        self.con_mlp = nn.Sequential(
            nn.Linear(41, 35),
            nn.ReLU(),
            nn.Linear(35, 35),
            nn.ReLU(),
            nn.Linear(35, 28),
            nn.ReLU()
            )
        
        self.fc = nn.Sequential(
            nn.Linear(40, 20),
            nn.ReLU(),
            nn.Linear(20, 10),
            nn.ReLU(),
            nn.Linear(10, 1),
            nn.Sigmoid()
            )
        
    def forward(self, ret, vol, turn, alloc, group):
        
        
        alloc = self.alloc_emb(alloc)
        group = self.group_emb(group)
        
        continuous = torch.cat([ret, vol, turn], dim=1)
        continuous = self.con_mlp(continuous)
        
        x = torch.cat([alloc, group, continuous], dim=1)
        
        return self.fc(x)


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
        for ret, vol, turn, alloc, group, y in train_loader:
            optimizer.zero_grad()
            prediction = model(ret, vol, turn, alloc, group)
            loss = loss_fn(prediction, y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.data.item()
            train_acc += calc_accuracy(prediction, y)
            
        train_loss_lst.append(train_loss/len(train_loader))
        train_acc_lst.append(train_acc/len(train_loader))
        
        
        val_loss = 0
        val_acc = 0
        model.eval()
        with torch.no_grad():
            for ret, vol, turn, alloc, group, y in val_loader:
                prediction = model(ret, vol, turn, alloc, group)
                loss = loss_fn(prediction, y)

                val_loss += loss.data.item()
                val_acc += calc_accuracy(prediction, y)

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

if __name__ == "__main__":
    
    X_train_raw = pd.read_csv("X_train.csv")[:]
    y_train_raw = pd.read_csv("y_train.csv")[:]
    y_train_raw["label"] = y_train_raw["target"] > 0
    
    X_test = pd.read_csv("X_test.csv")
    
    RET_COLS = [f"RET_{i+1}" for i in range(19, -1, -1)]
    VOL_COLS = [f"SIGNED_VOLUME_{i+1}" for i in range(19, -1, -1)]
    
    # data prep
    batch_size = 228
    X_train_raw = X_train_raw.drop(columns=["ROW_ID", "TS"])
    X_train_raw["ALLOCATION"] = X_train_raw["ALLOCATION"].str.extract(r"(\d+)").astype(int) - 1
    X_train_raw["GROUP"] = X_train_raw["GROUP"].astype(int) - 1
    
    X_train, X_val, y_train, y_val = train_test_split(X_train_raw, y_train_raw["label"])
    
    train_dataset = TimeSeriesData(
        X_train[RET_COLS], 
        X_train[VOL_COLS],
        X_train[["MEDIAN_DAILY_TURNOVER"]],
        X_train["ALLOCATION"],
        X_train["GROUP"], 
        y_train
        )
    train_data_loader = DataLoader(train_dataset, batch_size)
    
    val_dataset = TimeSeriesData(
        X_val[RET_COLS], 
        X_val[VOL_COLS],
        X_val[["MEDIAN_DAILY_TURNOVER"]],
        X_val["ALLOCATION"],
        X_val["GROUP"], 
        y_val
        )
    val_data_loader = DataLoader(val_dataset, batch_size)
    
    # training
    model = simpleNet()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    model, losses = train(model, optimizer, nn.BCELoss(),
                          train_data_loader, val_data_loader, 10)
    
    torch.save(model.state_dict(), "simplenet.pt")
    
    eva = evaluateModel(model, losses)
    eva.plot_loss(["train_loss", "val_loss"])
    eva.plot_loss(["train_acc", "val_acc"])









