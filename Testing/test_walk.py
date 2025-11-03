import torch
import time
import numpy as np

from humanoid_library import DQNAgent,QNetwork,HumanoidWalkEnv

env = HumanoidWalkEnv(render_mode="human")

# --- 1. Load environment in GUI mode ---
env = HumanoidWalkEnv(render_mode="human")   # GUI ON

# --- 2. Initialize agent ---
state_dim = env.state_dim
n_joints = env.n_joints
n_bins = 5  # same number of discrete bins used in training

agent = DQNAgent(state_dim=state_dim, n_joints=n_joints, n_bins=n_bins)

# --- 3. Load trained weights ---
agent.policy_net.load_state_dict(torch.load("trained_humanoid.pth"))
agent.policy_net.eval()
print("Loaded trained model")

# --- 4. Run a simulation ---
obs, _ = env.reset()
done = False

while not done:
    # Convert obs to tensor
    state_tensor = torch.FloatTensor(obs).unsqueeze(0)

    # Get discrete action bins → convert to torques
    with torch.no_grad():
        q_vals = agent.policy_net(state_tensor)
        best_bins = q_vals.argmax(dim=2).squeeze(0)
        torque_bins = torch.tensor([-1.0, -0.5, 0.0, 0.5, 1.0])
        torques = torque_bins[best_bins].numpy()

    # Step the environment
    obs, reward, terminated, truncated, info = env.step(torques)
    done = terminated or truncated

    # Small delay for human-viewable animation
    time.sleep(1/50)

env.close()