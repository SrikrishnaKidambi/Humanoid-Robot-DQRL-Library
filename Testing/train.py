import torch
import numpy as np
from humanoid_library import HumanoidWalkEnv
from humanoid_library import DQNAgent

# --- Pose library generation (dummy, replace later with real perception output) ---
def generate_initial_pose_library(num_poses=100, n_joints=8):
    """
    Creates a library of diverse initial poses.
    Replace this with pose_estimation.kinematicConversion output later.
    """
    pose_library = []
    for _ in range(num_poses):
        pose = np.random.uniform(-0.5, 0.5, size=(n_joints,))
        pose_library.append(pose)
    return pose_library


def train_dqn(num_episodes=500, max_steps=300, update_target_every=20):
    # === 1. Initialize environment and agent ===
    env = HumanoidWalkEnv(render_mode="human")
    state_dim = env.state_dim
    n_joints = env.n_joints
    n_bins = 5

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = DQNAgent(state_dim, n_joints, n_bins, device=device)

    # === 2. Pose library ===
    pose_library = generate_initial_pose_library(num_poses=200, n_joints=n_joints)

    all_rewards = []

    # === 3. Training loop ===
    for episode in range(num_episodes):
        # Random starting pose
        initial_pose = pose_library[np.random.randint(len(pose_library))]
        state, _ = env.reset(initial_pose=initial_pose)
        episode_reward = 0

        for step in range(max_steps):
            action = agent.select_action(state)
            next_state, reward, done, info = env.step(action)

            # Penalty for falling
            if done:
                reward -= 10.0

            agent.store_transition(state, action, reward, next_state, done)
            agent.train_step()

            state = next_state
            episode_reward += reward

            if done:
                break

        # Update target network periodically
        if (episode + 1) % update_target_every == 0:
            agent.update_target_network()

        all_rewards.append(episode_reward)
        print(f"[Episode {episode+1}/{num_episodes}] Reward: {episode_reward:.2f} | Epsilon: {agent.epsilon:.3f}")

    # === 4. Save trained model ===
    torch.save(agent.q_network.state_dict(), "./training/trained_dqn_agent.pth")
    print("✅ Training complete! Model saved to './training/trained_dqn_agent.pth'")

    return all_rewards


if __name__ == "__main__":
    train_dqn(num_episodes=1000, max_steps=500)
