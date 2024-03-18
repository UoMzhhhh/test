import os
import json
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define the model
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class TransformationNet(nn.Module):
    def __init__(self, num_features):
        super(TransformationNet, self).__init__()
        self.conv1 = ConvBlock(num_features, 16)
        self.conv2 = ConvBlock(16, 32)
        self.conv3 = ConvBlock(32, 256)
        self.global_pooling = nn.AdaptiveMaxPool1d(1)
        self.mlp1 = nn.Linear(256, 64)
        self.mlp2 = nn.Linear(64, 64)
        self.fc = nn.Linear(64, num_features * num_features)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.global_pooling(x)
        x = x.view(x.size(0), -1)
        x = self.mlp1(x)
        x = self.mlp2(x)
        x = self.fc(x)
        x = x.view(x.size(0), num_features, num_features)
        return x

class TransformationBlock(nn.Module):
    def __init__(self, num_features):
        super(TransformationBlock, self).__init__()
        self.transform = TransformationNet(num_features)

    def forward(self, x):
        transformed_features = self.transform(x)
        x_transformed = torch.bmm(x.transpose(1, 2), transformed_features)
        return x_transformed

class PointNet(nn.Module):
    def __init__(self, num_points, num_classes):
        super(PointNet, self).__init__()
        self.transform_block1 = TransformationBlock(6)
        self.conv1 = ConvBlock(6, 16)
        self.conv2 = ConvBlock(16, 32)
        self.conv3 = ConvBlock(32, 32)
        self.transform_block2 = TransformationBlock(32)
        self.conv4 = ConvBlock(32, 128)
        self.fc1 = nn.Linear(128 * num_points, 256)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x_transformed = self.transform_block1(x)
        x = self.conv1(x_transformed)
        x = self.conv2(x)
        x = self.conv3(x)
        x_transformed = self.transform_block2(x)
        x = self.conv4(x_transformed)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Define custom dataset
class CustomDataset(Dataset):
    def __init__(self, map_data, synt_data, label_data):
        self.map_data = map_data
        self.synt_data = synt_data
        self.label_data = label_data

    def __len__(self):
        return len(self.map_data)

    def __getitem__(self, idx):
        map_sample = self.map_data[idx]
        synt_sample = self.synt_data[idx]
        label = self.label_data[idx]
        return map_sample, synt_sample, label

# Define training function
def train_model(model, criterion, optimizer, train_loader, val_loader, num_epochs=10):
    train_losses = []
    val_losses = []
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for map_data, synt_data, labels in train_loader:
            map_data, synt_data, labels = map_data.to(device), synt_data.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(map_data)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * map_data.size(0)
        train_loss /= len(train_loader.dataset)
        train_losses.append(train_loss)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for map_data, synt_data, labels in val_loader:
                map_data, synt_data, labels = map_data.to(device), synt_data.to(device), labels.to(device)
                outputs = model(map_data)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * map_data.size(0)
        val_loss /= len(val_loader.dataset)
        val_losses.append(val_loss)

        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

    return model, train_losses, val_losses

# Assuming map_train, synt_train, label_train, map_val, synt_val, label_val are loaded

# Define model, criterion, optimizer
num_points = 2048*32
num_classes = 7
model = PointNet(num_points, num_classes).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# Create data loaders
train_dataset = CustomDataset(map_train, synt_train, label_train)
val_dataset = CustomDataset(map_val, synt_val, label_val)
train_loader = DataLoader(train_dataset, batch_size=5, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=5)

# Train the model
model, train_losses, val_losses = train_model(model, criterion, optimizer, train_loader, val_loader, num_epochs=1000)
