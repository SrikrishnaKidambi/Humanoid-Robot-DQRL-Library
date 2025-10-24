if __name__ == "__main__":
    from humanoid_library import (
    load_image, preprocess_image,
    PoseExtractor, select_main_skeleton_multiple, compute_joint_angles,
    HumanoidWalkEnv
)
    import numpy as np

    env = HumanoidWalkEnv(render_mode='human')
    
    # Example: Slight bending everywhere
    squat = np.array([0.3]*9, dtype=np.float32)

    obs, info = env.reset(initial_pose=squat)
    for step in range(150):
        action = np.random.uniform(-0.5, 0.5, size=(17,))
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Step {step}: reward {reward:.3f}")
        if terminated:
            print("Fell down. Resetting…")
            obs, info = env.reset(initial_pose=squat)

    env.close()
