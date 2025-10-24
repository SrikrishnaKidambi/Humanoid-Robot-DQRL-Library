import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pybullet
import pybullet_data
import os
import math
import time

NUM_JOINTS = 9

class HumanoidWalkEnv(gym.Env):
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 50}

    def __init__(self, render_mode=None):
            super().__init__()

            self.render_mode = render_mode

            if render_mode == "human":
                self.client = pybullet.connect(pybullet.GUI)
            else:
                self.client = pybullet.connect(pybullet.DIRECT)

            # 1. Set the search path
            data_path = pybullet_data.getDataPath()
            pybullet.setAdditionalSearchPath(data_path)
            pybullet.setGravity(0, 0, -9.81, physicsClientId=self.client)

            # 2. DEFINE THE CORRECT PATH
            # The file is in 'humanoid/', not 'mjcf/'
            self.model_path = os.path.join(data_path, "mjcf", "humanoid_symmetric.xml") # <-- THE FIX
            print(f"Model path set to: {self.model_path}")
            self.humanoid_id = None

            # Actuated joints count (PyBullet humanoid has 17 actuated joints)
            self.num_actions = 17
            self.action_space = spaces.Box(
                low=-1.0, high=1.0,
                shape=(self.num_actions,),
                dtype=np.float32
            )

            # Observations: joint angles + velocities + torso position + orientation
            self.obs_dim = 58 
            self.observation_space = spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(self.obs_dim,),
                dtype=np.float32
            )

            self.step_counter = 0
            self.max_steps = 1000


    def _load_robot(self):
        # 3. SIMPLIFIED AND CORRECT LOADING
        # This will now find the file because the path is correct 
        # and pybullet.setAdditionalSearchPath() was called in __init__
        safe_path = self.model_path.replace("\\", "/")
        try:
            print(f"ATTEMPTING TO LOAD MJCF FROM: {safe_path}")
            
            # Make sure you are using loadMJCF and getting the [0] index
            self.humanoid_id = pybullet.loadMJCF(safe_path, physicsClientId=self.client)[0] # <-- THE FIX
            
        except Exception as e:
            print(f"CRITICAL: Failed to load humanoid model from {self.model_path}")
            raise e


    def _get_actuated_joints(self):
        # This function is fine as-is, but will now work
        joint_info = [
            pybullet.getJointInfo(self.humanoid_id, i, physicsClientId=self.client)
            for i in range(pybullet.getNumJoints(self.humanoid_id, physicsClientId=self.client))
        ]
        self.actuated_joint_indices = [
            i for i, info in enumerate(joint_info)
            if info[2] in (pybullet.JOINT_REVOLUTE, pybullet.JOINT_PRISMATIC)
        ]
        
        # --- ADD THIS DEBUGGING CODE ---
        print("--- DEBUG: ACTUATED JOINT ORDER ---")
        for i in self.actuated_joint_indices:
            name = pybullet.getJointInfo(self.humanoid_id, i)[1].decode('UTF-8')
            print(f"Index {i}: {name}")
        print("-----------------------------------")
        # --- END DEBUGGING CODE ---

    #absolute mapping to urdf angles and our returned vetor angles (9 angles between various joints)
    # def _apply_initial_pose(self, initial_pose):
    #     if initial_pose is not None:
    #         if initial_pose.shape[0] != NUM_JOINTS:
    #             raise ValueError(
    #                 f"Expected pose vector of length {NUM_JOINTS}, got {initial_pose.shape[0]}"
    #             )

    #         # --- THIS IS THE NEW MAPPING ---
            
    #         # This dictionary maps our 9-angle vector to the simulation's joint indices.
    #         # 'vector_index': sim_joint_index
    #         joint_mapping = {
    #             # Angle 0 (R Hip) -> 'right_hip_x' (Index 5)
    #             0: 5,
    #             # Angle 1 (R Knee) -> 'right_knee' (Index 9)
    #             1: 9,
    #             # Angle 2 (L Hip) -> 'left_hip_x' (Index 12)
    #             2: 12,
    #             # Angle 3 (L Knee) -> 'left_knee' (Index 16)
    #             3: 16,
                
    #             # --- THIS IS THE FIX ---
    #             # Angle 4 (R Shoulder) -> 'right_shoulder2' (Index 20)
    #             4: 20,
    #             # --- END FIX ---

    #             # Angle 5 (R Elbow) -> 'right_elbow' (Index 22)
    #             5: 22,
                
    #             # --- THIS IS THE FIX ---
    #             # Angle 6 (L Shoulder) -> 'left_shoulder2' (Index 25)
    #             6: 25,
    #             # --- END FIX ---
                
    #             # Angle 7 (L Elbow) -> 'left_elbow' (Index 27)
    #             7: 27,
    #             # Angle 8 (Spine) -> 'abdomen_y' (Index 1)
    #             8: 1
    #         }

    #         # Now, loop through our 9 angles and apply them to the correct joints
    #         for vector_index, joint_index in joint_mapping.items():
                
    #             # Get the angle value from our pose vector
    #             target_angle = initial_pose[vector_index]

    #             # Apply it to the correct joint in the simulation
    #             pybullet.resetJointState(
    #                 self.humanoid_id,
    #                 joint_index,
    #                 target_angle,
    #                 physicsClientId=self.client
    #             )

    # In humanoid_env.py

    def _apply_initial_pose(self, initial_pose):
        if initial_pose is not None:
            if initial_pose.shape[0] != NUM_JOINTS:
                raise ValueError(
                    f"Expected pose vector of length {NUM_JOINTS}, got {initial_pose.shape[0]}"
                )

            # --- FINAL MAPPING (9-AXIS) ---
            # We map our 2D angles to the correct 3D joint axis (mostly 'y' for forward/back)
            joint_mapping = {
                # Angle 0 (R Hip) -> 'right_hip_y' (Index 7)
                0: 7,
                # Angle 1 (R Knee) -> 'right_knee' (Index 9)
                1: 9,
                # Angle 2 (L Hip) -> 'left_hip_y' (Index 14)
                2: 14,
                # Angle 3 (L Knee) -> 'left_knee' (Index 16)
                3: 16,
                # Angle 4 (R Shoulder) -> 'right_shoulder1' (Index 19)
                # (We'll use shoulder1 for the main swing)
                4: 19,
                # Angle 5 (R Elbow) -> 'right_elbow' (Index 22)
                5: 22,
                # Angle 6 (L Shoulder) -> 'left_shoulder1' (Index 24)
                6: 24,
                # Angle 7 (L Elbow) -> 'left_elbow' (Index 27)
                7: 27,
                # Angle 8 (Spine) -> 'abdomen_y' (Index 1)
                8: 1
            }
            
            # Loop through our 9 angles and apply them
            for vector_index, joint_index in joint_mapping.items():
                
                target_angle = initial_pose[vector_index]

                pybullet.resetJointState(
                    self.humanoid_id,
                    joint_index,
                    target_angle,
                    physicsClientId=self.client
                )


    def _get_obs(self):
        # Torso position + orientation (quaternion)
        pos, orn = pybullet.getBasePositionAndOrientation(self.humanoid_id)

        # Joint angles + velocities
        joints = []
        for j in self.actuated_joint_indices:
            state = pybullet.getJointState(self.humanoid_id, j)
            joints.extend([state[0], state[1]]) # position, velocity

        obs = np.array(list(pos) + list(orn) + joints, dtype=np.float32)
        return obs


    def reset(self, seed=None, options=None, initial_pose=None):
        super().reset(seed=seed)
        self.step_counter = 0

        pybullet.resetSimulation(physicsClientId=self.client)
        pybullet.setGravity(0, 0, -9.81, physicsClientId=self.client)
        pybullet.setTimeStep(1/240, physicsClientId=self.client)

        pybullet.loadURDF("plane.urdf", [0,0,0], physicsClientId=self.client)

        self._load_robot()
        self._get_actuated_joints()
        self._apply_initial_pose(initial_pose)

        # Let the robot stabilize a bit
        for _ in range(10):
            pybullet.stepSimulation()

        obs = self._get_obs()
        info = {}
        return obs, info


    def step(self, action):
        # Clip actions to safe torques
        action = np.clip(action, -1, 1)

        pybullet.setJointMotorControlArray(
            bodyUniqueId=self.humanoid_id,
            jointIndices=self.actuated_joint_indices[:self.num_actions],
            controlMode=pybullet.TORQUE_CONTROL,
            forces=action
        )

        pybullet.stepSimulation()
        self.step_counter += 1

        obs = self._get_obs()
        reward = self._compute_reward(obs)
        terminated = self._check_termination(obs)
        truncated = self.step_counter >= self.max_steps
        info = {}

        if self.render_mode == "human":
            time.sleep(1/self.metadata["render_fps"])

        return obs, reward, terminated, truncated, info


    def _compute_reward(self, obs):
        # Forward velocity reward from base linear velocity
        vel = pybullet.getBaseVelocity(self.humanoid_id)[0][0]
        r_vel = max(vel, 0)

        # Alive bonus
        r_live = 0.05

        # Energy penalty
        torque_penalty = 0.001 * np.sum(np.square(obs[10:10+self.num_actions]))

        return r_vel + r_live - torque_penalty


    def _check_termination(self, obs):
        z = obs[2]  # torso height
        if z < 0.7:
            return True
        return False


    def close(self):
        pybullet.disconnect(self.client)
