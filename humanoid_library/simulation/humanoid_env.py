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

        data_path = pybullet_data.getDataPath()
        pybullet.setAdditionalSearchPath(data_path)
        pybullet.setGravity(0, 0, -9.81, physicsClientId=self.client)

        self.model_path = os.path.join(data_path, "mjcf", "humanoid_symmetric.xml") 
        print(f"Model path set to: {self.model_path}")
        self.humanoid_id = None

        # --- Action and observation spaces ---
        self.num_actions = 17
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.num_actions,), dtype=np.float32)
        self.obs_dim = 58
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32)

        # --- Add these attributes ---
        self.state_dim = self.obs_dim
        self.n_joints = NUM_JOINTS

        self.step_counter = 0
        self.max_steps = 1000


    def _load_robot(self):
        safe_path = self.model_path.replace("\\", "/")
        try:
            # print(f"ATTEMPTING TO LOAD MJCF FROM: {safe_path}")
            
            self.humanoid_id = pybullet.loadMJCF(safe_path, physicsClientId=self.client)[0] 
            
        except Exception as e:
            print(f"CRITICAL: Failed to load humanoid model from {self.model_path}")
            raise e


    def _get_actuated_joints(self):
        joint_info = [
            pybullet.getJointInfo(self.humanoid_id, i, physicsClientId=self.client)
            for i in range(pybullet.getNumJoints(self.humanoid_id, physicsClientId=self.client))
        ]

        # 🔧 Instead of controlling all 17, pick the 9 corresponding joints
        self.control_joint_indices = [
            7,   # right_hip_y
            9,   # right_knee
            14,  # left_hip_y
            16,  # left_knee
            19,  # right_shoulder1
            22,  # right_elbow
            24,  # left_shoulder1
            27,  # left_elbow
            1    # abdomen_y (spine)
        ]

        self.num_actions = len(self.control_joint_indices)  # 9
        self.actuated_joint_indices = self.control_joint_indices
        
        
        # p---------------------------------")
        

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


    def _apply_initial_pose(self, initial_pose):
        if initial_pose is not None:
            if initial_pose.shape[0] != NUM_JOINTS:
                raise ValueError(
                    f"Expected pose vector of length {NUM_JOINTS}, got {initial_pose.shape[0]}"
                )

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
        pos, orn = pybullet.getBasePositionAndOrientation(self.humanoid_id)
        joints = []
        for j in self.actuated_joint_indices:
            state = pybullet.getJointState(self.humanoid_id, j)
            joints.extend([state[0], state[1]])  # position, velocity
    
        obs = np.array(list(pos) + list(orn) + joints, dtype=np.float32)
    
        # Pad to match 41 dims if shorter
        if len(obs) < 41:
            obs = np.pad(obs, (0, 41 - len(obs)), mode='constant')
    
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
        # Clip and match action dimension
        action = np.clip(action, -1, 1)

        if len(action) != len(self.control_joint_indices):
            raise ValueError(
                f"Action length mismatch: expected {len(self.control_joint_indices)}, got {len(action)}"
            )

        # 🔧 Apply torques to only these 9 controlled joints
        pybullet.setJointMotorControlArray(
            bodyUniqueId=self.humanoid_id,
            jointIndices=self.control_joint_indices,
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
            time.sleep(1 / self.metadata["render_fps"])

        return obs, reward, terminated, truncated, info


    def _compute_reward(self, obs):
        """
        Compute reward based on current state (obs) and last applied torques

        Reward formula:
            R_t = w_vel * r_vel + w_live * r_live - w_energy * r_energy
        """
        
        # Extract simulation data
        base_lin_vel, base_ang_vel = pybullet.getBaseVelocity(self.humanoid_id,physicsClientId = self.client)
        pos, orn = pybullet.getBasePositionAndOrientation(self.humanoid_id, physicsClientId=self.client)
        z_height = pos[2]

        # Forward velocity reward (positive x direction)
        r_vel = max(base_lin_vel[0],0.0) 
        # so if move +ve x then the x direction vel is positive x reward but if going in -ve x direction then the vel is -ve so reward is 0

        height_threshold = 0.7
        r_live = 1.0 if z_height > height_threshold else 0.0
        # gives +ve 1 reward if height > threshold else the reward is 0 and this is live reward

        r_energy = np.sum(np.square(getattr(self, 'last_action', np.zeros(self.num_actions))))
        # here it penalizes for large or abrupt torques for smooth and efficient motiion
        # self.last_action has torques applied in prev step
        # we do np.square so large torques get more penality than smaller ones
        # so getattr ensures that if last_action don't exist then it sets all the action values to 0
        
        # defining the weights
        w_vel = 1.0 # makes forward walking the main objective 
        w_live = 0.2 # gives a small incentive for staying upright
        w_energy = 0.001 # small penalty to encourage efficiency

        # final reward
        reward = (w_vel * r_vel) + (w_live * r_live) - (w_energy * r_energy)

        if self._check_termination(obs):
            reward -= 5.0

        return float(reward)

    def _check_termination(self, obs):
        z = obs[2]  # torso height
        if z < 0.7:
            return True
        return False


    def close(self):
        pybullet.disconnect(self.client)
