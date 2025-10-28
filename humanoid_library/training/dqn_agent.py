import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class QNetwork(nn.Module):
    def __init__(self, state_dim, n_joints, n_bins, hidden_dims=(256, 256)):
        super(QNetwork, self).__init__()
        self.n_joints = n_joints
        self.n_bins = n_bins

        self.fc1 = nn.Linear(state_dim, hidden_dims[0])
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.out = nn.Linear(hidden_dims[1], n_joints * n_bins)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.out(x)
        return x.view(-1, self.n_joints, self.n_bins)  # shape: [batch, joints, bins]

