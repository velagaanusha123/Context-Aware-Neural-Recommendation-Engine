import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
import os
from tqdm import tqdm

# --------------------------------------------------------------------------- #
# Neural Collaborative Filtering (NCF) Model
# --------------------------------------------------------------------------- #

class NCFModel(nn.Module):
    def __init__(self, n_users, n_items, embedding_dim=64):
        super(NCFModel, self).__init__()
        
        # Embedding layers
        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        self.item_embedding = nn.Embedding(n_items, embedding_dim)
        
        # Dense layers (MLP)
        self.fc_layers = nn.Sequential(
            nn.Linear(embedding_dim * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
    def forward(self, user_indices, item_indices):
        user_embeds = self.user_embedding(user_indices)
        item_embeds = self.item_embedding(item_indices)
        
        # Concatenate user and item embeddings
        x = torch.cat([user_embeds, item_embeds], dim=-1)
        
        # Pass through dense layers
        output = self.fc_layers(x)
        return output.view(-1)

# --------------------------------------------------------------------------- #
# Dataset Loader
# --------------------------------------------------------------------------- #

class RecommendationDataset(Dataset):
    def __init__(self, user_indices, item_indices, labels):
        self.user_indices = torch.LongTensor(user_indices)
        self.item_indices = torch.LongTensor(item_indices)
        self.labels = torch.FloatTensor(labels)
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.user_indices[idx], self.item_indices[idx], self.labels[idx]

# --------------------------------------------------------------------------- #
# Training Logic
# --------------------------------------------------------------------------- #

def train_neural_model(csv_path, model_save_path):
    print(f"[Neural] Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # Use only essential columns
    users = df['user_idx'].values
    items = df['item_idx'].values
    labels = df['interaction'].values
    
    n_users = users.max() + 1
    n_items = items.max() + 1
    
    # Create dataset and loader
    dataset = RecommendationDataset(users, items, labels)
    loader = DataLoader(dataset, batch_size=256, shuffle=True)
    
    # Initialize model
    model = NCFModel(n_users, n_items)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Train for a few epochs
    print("[Neural] Starting training...")
    model.train()
    epochs = 3
    for epoch in range(epochs):
        total_loss = 0
        for u, i, l in tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}"):
            optimizer.zero_grad()
            preds = model(u, i)
            loss = criterion(preds, l)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1} Loss: {total_loss/len(loader):.4f}")
    
    # Save model
    print(f"[Neural] Saving model to {model_save_path}")
    torch.save({
        'model_state_dict': model.state_dict(),
        'n_users': n_users,
        'n_items': n_items
    }, model_save_path)
    
    return n_users, n_items

if __name__ == "__main__":
    # Test/Run training script
    BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_PATH = os.path.join(BASE, "Data", "Processed data", "model_ready_recommendation_dataset.csv")
    SAVE_PATH = os.path.join(BASE, "model", "ncf_model.pth")
    
    if not os.path.exists(os.path.dirname(SAVE_PATH)):
        os.makedirs(os.path.dirname(SAVE_PATH))
        
    train_neural_model(DATA_PATH, SAVE_PATH)

