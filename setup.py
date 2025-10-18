from setuptools import setup, find_packages

setup(
    name="humanoid_walk",  # name of your library
    version="0.1.0",        # initial version
    author="ISL_Team20",
    author_email="cs23b0{58,59,60}@iittp.ac.in",
    description="A Python library for humanoid locomotion from image-defined initial poses using Deep Reinforcement Learning",
    packages=find_packages(),  # automatically finds all packages inside humanoid_walk/
    install_requires=[
        "gymnasium",
        "pybullet",
        "mediapipe",
        "opencv-python",
        "numpy",
        "torch",
        "stable-baselines3",
        "matplotlib"
    ],
    python_requires='>=3.8',
)