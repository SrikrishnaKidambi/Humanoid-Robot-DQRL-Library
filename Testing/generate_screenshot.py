
import cv2
import numpy as np
import os
import pybullet
from humanoid_library.pose_estimation import imgAcquisition, keyPointExtraction, kinematicConversion
from humanoid_library.simulation import humanoid_env

# This script runs the full pipeline on a test image to generate visual results for the report.

# Ensure the output directory exists
output_dir = "Testing"
os.makedirs(output_dir, exist_ok=True)

# --- 1. Run the Pose Estimation Pipeline ---
print("Loading image...")
# Use "People.jpg" to demonstrate the multi-person selection capability
image_path = os.path.join("Testing", "People.jpg")
img = imgAcquisition.load_image(image_path)

print("Extracting skeleton...")
extractor = keyPointExtraction.PoseExtractor()
# Save the skeleton overlay for the report
skeleton_output_path = os.path.join(output_dir, "output_pose_for_report.jpg")
skeleton = kinematicConversion.select_main_skeleton_multiple(extractor, img, skeleton_output_path, iterations=10)

if skeleton is None:
    print("Could not detect a skeleton. Exiting.")
    exit()

print("Computing joint angles...")
initial_pose_angles = kinematicConversion.compute_joint_angles(skeleton)

# --- 2. Generate Simulation Screenshot ---
print("Initializing simulation environment in 'rgb_array' mode...")
# Use 'rgb_array' to render without a GUI window
env = humanoid_env.HumanoidWalkEnv(render_mode='rgb_array')

print("Resetting environment with initial pose...")
obs, info = env.reset(initial_pose=initial_pose_angles)

# Let the simulation settle for a few steps so the humanoid is stable
for _ in range(20):
    pybullet.stepSimulation(env.client)

print("Positioning camera and rendering screenshot...")
# Use the humanoid's position to center the camera
base_pos, _ = pybullet.getBasePositionAndOrientation(env.humanoid_id)

view_matrix = pybullet.computeViewMatrixFromYawPitchRoll(
    cameraTargetPosition=base_pos,
    distance=2.5,
    yaw=60,
    pitch=-20,
    roll=0,
    upAxisIndex=2
)

proj_matrix = pybullet.computeProjectionMatrixFOV(
    fov=60,
    aspect=1024/768,
    nearVal=0.1,
    farVal=100.0
)

# Render the image
width, height, rgb_img, depth_img, seg_img = pybullet.getCameraImage(
    width=1024,
    height=768,
    viewMatrix=view_matrix,
    projectionMatrix=proj_matrix,
    renderer=pybullet.ER_BULLET_HARDWARE_OPENGL
)

env.close()

# Process the image and save it
# PyBullet returns an RGBA image, so we convert it to RGB for saving
rgb_array = np.array(rgb_img, dtype=np.uint8)[:, :, :3]

screenshot_path = os.path.join(output_dir, "humanoid_screenshot.png")
print(f"Saving screenshot to {screenshot_path}")
# Convert RGB to BGR for OpenCV's imwrite function
cv2.imwrite(screenshot_path, cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR))

print("Script finished. Visual assets are ready for the report.")
