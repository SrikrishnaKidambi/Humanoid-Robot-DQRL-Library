import random
from collections import deque
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch import nn


class QNetwork(nn.Module):
    def __init__(self, state_dim, n_joints, n_bins):
        super(QNetwork, self).__init__()
        self.n_joints = n_joints
        self.n_bins = n_bins
        self.output_dim = n_joints * n_bins

        self.net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, self.output_dim)
        )

    def forward(self, state):
        x = self.net(state)
        return x.view(-1, self.n_joints, self.n_bins)  # shape: [batch, joints, bins]


class DQNAgent:
    def __init__(
        self,
        state_dim,
        n_joints,
        n_bins,
        lr=1e-3,
        gamma=0.99,
        epsilon=1.0,
        epsilon_decay=0.995,
        epsilon_min=0.05,
        memory_size=10000,
        batch_size=64,
        device=None
    ):
        self.state_dim = state_dim
        self.n_joints = n_joints
        self.n_bins = n_bins
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size

        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Torque bins (discrete actions)
        self.torque_bins = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
        assert n_bins == len(self.torque_bins), "n_bins must match number of torque bins."

        # Q-networks
        self.q_network = QNetwork(state_dim, n_joints, n_bins).to(self.device)
        self.target_network = QNetwork(state_dim, n_joints, n_bins).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())

        # Optimizer and replay memory
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.memory = deque(maxlen=memory_size)

    def select_action(self, state):
        """Select action with epsilon-greedy strategy."""
        if np.random.rand() < self.epsilon:
            # Random torques per joint
            return np.random.choice(self.torque_bins, size=self.n_joints)

        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_network(state)[0].cpu().numpy()  # shape: [joints, bins]

        action_indices = np.argmax(q_values, axis=1)
        # action = self.torque_bins[action_indices]
        action = np.random.uniform(-0.5, 0.5, size=(9,))
        return action

    def store_transition(self, state, action, reward, next_state, done):
        """Store experience in replay buffer."""
        # Convert continuous actions to bin indices
        action_indices = np.array([np.argmin(np.abs(self.torque_bins - a)) for a in action])
        self.memory.append((state, action_indices, reward, next_state, done))

    def train_step(self):
        """Perform one training update."""
        if len(self.memory) < self.batch_size:
            return

        # Sample batch
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)  # shape: [B, n_joints]
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        # Compute current Q values
        q_pred = self.q_network(states)  # [B, n_joints, n_bins]
        q_pred = q_pred.gather(2, actions.unsqueeze(2)).squeeze(2).mean(dim=1, keepdim=True)

        # Compute target Q values
        with torch.no_grad():
            q_next = self.target_network(next_states)  # [B, n_joints, n_bins]
            q_next_max = q_next.max(2)[0].mean(dim=1, keepdim=True)
            target_q = rewards + (1 - dones) * self.gamma * q_next_max

        # Loss and backpropagation
        loss = F.mse_loss(q_pred, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Epsilon decay
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def update_target_network(self):
        """Hard update of target network weights."""
        self.target_network.load_state_dict(self.q_network.state_dict())
