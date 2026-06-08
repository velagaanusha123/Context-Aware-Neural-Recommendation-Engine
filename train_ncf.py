
import torch
from ncf_model import NCF

def train(model, data_loader, epochs=5):
    loss_fn = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(epochs):
        total_loss = 0
        for user, item, label in data_loader:
            pred = model(user, item).squeeze()
            loss = loss_fn(pred, label.float())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}: Loss {total_loss:.4f}")
