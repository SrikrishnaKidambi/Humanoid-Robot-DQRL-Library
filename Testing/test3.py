import numpy as np
import os
from humanoid_library import (
    # Module 1 functions
    load_image, 
    preprocess_image,
    PoseExtractor, 
    select_main_skeleton_multiple, 
    compute_joint_angles,
    
    # Module 2 class
    HumanoidWalkEnv
)

if __name__ == "__main__":
    
    # --- MODULE 1: POSE ESTIMATION ---
    script_dir = os.path.dirname(__file__)
    # 1. Define image paths
    image_path = os.path.join(script_dir, "People.jpg")
    save_path = os.path.join(script_dir, "Output_Pose.jpg")
    
    print(f"Loading image from: {image_path}")
    image = load_image(image_path)
    image_r = preprocess_image(image, (640, 480))

    # 2. Extract skeleton
    extractor = PoseExtractor()
    main_skeleton = select_main_skeleton_multiple(extractor, image_r, save_path)

    # 3. Compute joint angles
    if main_skeleton is not None:
        initial_pose_vector = compute_joint_angles(main_skeleton)
        print(f"Successfully computed joint angles: {initial_pose_vector}")

        # --- MODULE 2: SIMULATION ---
        
        print("Launching simulation with extracted pose...")
        env = HumanoidWalkEnv(render_mode='human')
        
        # --- THE BRIDGE: Pass the vector ---
        # Instead of the 'squat' array, we use the vector from Module 1
        obs, info = env.reset(initial_pose=initial_pose_vector)
        # ------------------------------------


        for step in range(300): # Run for a bit longer to see
            action = np.random.uniform(-0.5, 0.5, size=(17,))
            obs, reward, terminated, truncated, info = env.step(action)
            
            if terminated:
                print(f"Step {step}: Fell down. Resetting simulation to the image pose.")
                obs, info = env.reset(initial_pose=initial_pose_vector)

        env.close()
        print("Simulation finished.")

    else:
        print(f"Could not find a person in the image: {image_path}")