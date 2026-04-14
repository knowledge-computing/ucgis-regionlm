import os, sys
import argparse
from pathlib import Path

import pandas as pd
import ast

import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.optim as optim

from dimension_reducer._base import DimensionReducerBase
from utils import const


class AutoencoderReducer(DimensionReducerBase):
    def __init__(self):
        super(AutoencoderReducer, self).__init__()

    def fit_transform(self, in_csv_file_path, out_csv_file_path, dimension=64, epoch = 300):
        self.in_csv_file_path = in_csv_file_path
        self.out_csv_file_path = out_csv_file_path
        self.dimension = dimension 
        self.epoch = epoch

        self.df = pd.read_csv(self.in_csv_file_path)
    
        embdf = self.df[const.spabert_emb_field_name].apply(lambda x: ast.literal_eval(x))
        embdf = pd.DataFrame(embdf.tolist())

        x_data = torch.tensor(embdf.values, dtype=torch.float32)
       
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        x_data_tensor = x_data.to(device)
        x_dataset = TensorDataset(x_data_tensor, x_data_tensor)

        x_loader = DataLoader(x_dataset, batch_size=256, shuffle=True)
       
        autoencoder = Autoencoder(in_shape=x_data.shape[1], enc_shape=self.dimension).to(device)
        criterion = nn.MSELoss()
        optimizer = optim.Adadelta(autoencoder.parameters())
        train(autoencoder, criterion, optimizer, self.epoch, x_loader, device)
        encoded_data = autoencoder.encode(x_data_tensor)
        encoded_data = encoded_data.cpu()

        self.df[const.spabert_emb_enc_field_name] = encoded_data.detach().numpy().tolist()
        self.df.to_csv(self.out_csv_file_path, index=False)
        return self.df


class Autoencoder(nn.Module):
    def __init__(self, in_shape, enc_shape):
        super(Autoencoder, self).__init__()
        
        self.encode = nn.Sequential(
            nn.Linear(in_shape, 128),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Linear(64, enc_shape),
        )
        
        self.decode = nn.Sequential(
            nn.BatchNorm1d(enc_shape),
            nn.Linear(enc_shape, 64),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Linear(64, 128),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Linear(128, in_shape)
        )

    def forward(self, x):
        encoded = self.encode(x)
        decoded = self.decode(encoded)
        return decoded


def train(model, criterion, optimizer, n_epochs, train_loader, device):
    model.train()
    for epoch in range(n_epochs):
        total_loss = 0
        for batch_data, _ in train_loader:
            batch_data = batch_data.to(device)
            optimizer.zero_grad()
            outputs = model(batch_data)
            loss = criterion(outputs, batch_data)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch [{epoch+1}/{n_epochs}], Loss: {total_loss/len(train_loader):.4f}")