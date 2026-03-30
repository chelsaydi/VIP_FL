import torch
import torch.nn as nn

class TMDNet(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_classes=4):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        return self.net(x)
