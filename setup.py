from setuptools import setup, find_packages

setup(
    name="humanoid_library",  # name of your library
    version="0.1.0",        # initial version
    description="A Python library for humanoid locomotion from image-defined initial poses using Deep Reinforcement Learning",
    packages=find_packages(),  # automatically finds all packages inside humanoid_walk/
    install_requires=[
        'opencv-python',
        'numpy',
        'Pillow',
        'mediapipe'
    ],
    python_requires='>=3.8',
)