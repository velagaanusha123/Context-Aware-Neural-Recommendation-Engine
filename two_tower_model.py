
import torch
import torch.nn as nn

class TwoTower(nn.Module):
    def __init__(self, num_users, num_items, dim=64):
        super().__init__()
        self.user_net = nn.Sequential(
            nn.Embedding(num_users, dim),
            nn.Linear(dim, dim)
        )

        self.item_net = nn.Sequential(
            nn.Embedding(num_items, dim),
            nn.Linear(dim, dim)
        )

    def forward(self, user, item):
        u = self.user_net(user)
        i = self.item_net(item)
        return (u * i).sum(dim=1)
