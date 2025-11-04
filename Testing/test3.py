import os
import torch
import numpy as np
from humanoid_library import (
    load_image, 
    preprocess_image,
    PoseExtractor, 
    select_main_skeleton_multiple, 
    compute_joint_angles,
    HumanoidWalkEnv,
    DQNAgent
)

if __name__ == "__main__":
    # --- MODULE 1: POSE ESTIMATION ---
    script_dir = os.path.dirname(__file__)
    image_path = os.path.join(script_dir, "img1.jpg")
    save_path = os.path.join(script_dir, "out_img1.jpg")
    
    print(f"Loading image from: {image_path}")
    image = load_image(image_path)
    image_r = preprocess_image(image, (640, 480))

    # Extract skeleton
    extractor = PoseExtractor()
    main_skeleton = select_main_skeleton_multiple(extractor, image_r, save_path)

    initial_pose_vector = None
    if main_skeleton is not None:
        initial_pose_vector = compute_joint_angles(main_skeleton)
        print(f"Computed joint angles: {initial_pose_vector}")
    else:
        print("No skeleton detected. Using default upright pose.")

    # --- MODULE 2: SIMULATION WITH TRAINED DQN ---
    print("Launching simulation with DQN-controlled humanoid...")
    env = HumanoidWalkEnv(render_mode='human')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    state_dim = 41
    n_joints = 9
    n_bins = 5

    # Initialize the agent and load trained weights
    agent = DQNAgent(state_dim=state_dim, n_joints=n_joints, n_bins=n_bins, device=device)
    model_path = os.path.join(script_dir, "dqn_policy.pth")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Trained DQN model not found at {model_path}")

    print("Loading trained DQN model...")
    agent.policy_net.load_state_dict(torch.load(model_path, map_location=device))
    agent.policy_net.eval()
    print("Model loaded successfully!")

    # --- RUN SIMULATION ---
    obs, info = env.reset(initial_pose=initial_pose_vector)
    print("Starting humanoid walking simulation...")

    # Map DQN 9-joint outputs → first 9 actuated joints in env
    controlled_joint_indices = env.actuated_joint_indices[:n_joints]  # safest mapping

    for step in range(1000):
        state_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            torques_9_tensor = agent.policy_net.get_discrete_action(state_tensor)
            torques_9 = torques_9_tensor.squeeze(0).cpu().numpy()  # shape -> (9,) or (9, n_bins)

            # If output is 2D (bins), pick max bin per joint
            if torques_9.ndim == 2:
                torques_9 = torques_9.argmax(axis=1)

        full_action = np.zeros(env.num_actions, dtype=np.float32)

        # Assign torques safely to actuated joints
        for i, joint_idx in enumerate(controlled_joint_indices):
            print(f"{i} and {joint_idx}")
            full_action[joint_idx-1] = torques_9[i]

        obs, reward, terminated, truncated, info = env.step(full_action)

        if terminated or truncated:
            print(f"Step {step}: Episode ended. Resetting to initial pose.")
            obs, info = env.reset(initial_pose=initial_pose_vector)

    env.close()
    print("Simulation finished successfully.")
