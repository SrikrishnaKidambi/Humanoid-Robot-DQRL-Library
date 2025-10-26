# Humanoid-Robot-DQRL-Library

### This project provides a library for Deep Reinforcement Learning with a humanoid robot.

It includes:
- Module 1: Pose estimation from camera images
- Module 2: PyBullet physics simulation environment

# Setup Instructions:

1) Clone the Repository:
```
git clone https://github.com/your-username/Humanoid-Robot-DQRL-Library.git
cd Humanoid-Robot-DQRL-Library
```

2) Create a Virtual Environment:
**Python Version Warning: This project requires mediapipe. It does not support Python 3.13 or newer.**
Use Python 3.12, 3.11, or 3.10.
Example: `py -3.12 -m venv venv`

3) Activate the Environment:
Windows: `.venv\Scripts\activate`
Linux/macOS: `source venv/bin/activate`

4) Install Dependencies:
`pip install numpy ultralytics opencv-python mediapipe gymnasium pybullet ultralytics`

5) Install custom library ( humanoid_library )
From the project root, run this command:
`pip install -e .`
This makes sure our humanoid_library is visible when called through other files and testing scripts.