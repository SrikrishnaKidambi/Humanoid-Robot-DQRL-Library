import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    def __init__(self,state_dim, n_joints, n_bins):
        """
        Initialize the Q Network.

        Args:
            state_dim (int) : Dimension of the input state vector
            n_joints(int): Number of controllable joints.
            n_bins(int): Number of discrete torque bins per joint
        """
        
        super(QNetwork,self).__init__()

        self.n_joints = n_joints
        self.n_bins = n_bins

        # here we are giving a head for each joint not the traditional dqn learning stuff
        # So for each joint we learn the best discrete torque seperately and then pick the torque that maximize the reward for each head/joint
        # So we need not to define action space of size 5^8 rather we just define aciton space of 5*8 for 8 joints and 5 bins.
        self.total_actions = n_joints * n_bins
        # defining the layers
        self.fc1 = nn.Linear(state_dim,256) #first hidden layer
        self.fc2 = nn.Linear(256,256) #second hidden layer
        self.fc3 = nn.Linear(256,512) #third hidden layer
        self.out = nn.Linear(512,self.total_actions) # output layer

        # discrete torque bins
        self.torque_bins = torch.tensor([-1.0, -0.5, 0.0, 0.5, 1.0])

    def forward(self,state):
        """
        Forward pass through the Q-network

        Args:
            state (torch.Tensor): Input state tensor of shape(batch_size,state_dim)
        Returns:
            torch.Tensor: Q-values for each action of shape (batch_size,action_dim)
        """

        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        q_values = self.out(x)

        # reshape to [batch_size,n_joints,n_bins]
        q_values = q_values.view(-1,self.n_joints,self.n_bins)
        return q_values # here q_values are the rewards for various actions that are possible in the state(total_actions)
    
    def get_discrete_action(self,state):
        """
        Given a state, select the best discrete torque bin for each joint.
        Returns a torch.Tensor: continuous torque vector [n_joints]
        """
        with torch.no_grad():
            q_values = self.forward(state) # [we get an array of [batch_size,n_joints,n_torque_bins]]
            best_bin_indices = q_values.argmax(dim=2).squeeze(0) # [n_joints]
            continuous_torques = self.torque_bins[best_bin_indices]
        return continuous_torques # size is [n_joints]
    

#just for internal testing
if __name__ == "__main__":
    state_dim = 24
    n_joints = 8
    n_bins = 5

    model = QNetwork(state_dim,n_joints,n_bins)
    sample_state = torch.randn(1,state_dim)

    q_vals = model(sample_state)
    print("Q_values shape:",q_vals.shape)
    print("Q_values: ",q_vals) # reward values for each joint for each torque (head for each joint)

    action = model.get_discrete_action(sample_state)
    print("Selected torques: ",action) # length 8 tensor