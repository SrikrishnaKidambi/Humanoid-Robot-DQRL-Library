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
        self.num_actions = 9  # We control 9 joints
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.num_actions,), dtype=np.float32)
        self.obs_dim = 25  # 3 pos + 4 orn + 9*2 joints = 25
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32)
        self.state_dim = self.obs_dim
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
        # self.control_joint_indices = [
        #     7,   # right_hip_y
        #     9,   # right_knee
        #     14,  # left_hip_y
        #     16,  # left_knee
        #     19,  # right_shoulder1
        #     22,  # right_elbow
        #     24,  # left_shoulder1
        #     27,  # left_elbow
        #     1    # abdomen_y (spine)
        # ]
        self.control_joint_indices = [
            7,   # right_hip_y    <- FORWARD/BACK leg swing
            9,   # right_knee
            14,  # left_hip_y     <- FORWARD/BACK leg swing
            16,  # left_knee
            19,  # right_shoulder1 <- MAIN arm swing
            22,  # right_elbow
            24,  # left_shoulder1  <- MAIN arm swing
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

            # joint_mapping = {
            #     # Angle 0 (R Hip) -> 'right_hip_y' (Index 7)
            #     0: 7,
            #     # Angle 1 (R Knee) -> 'right_knee' (Index 9)
            #     1: 9,
            #     # Angle 2 (L Hip) -> 'left_hip_y' (Index 14)
            #     2: 14,
            #     # Angle 3 (L Knee) -> 'left_knee' (Index 16)
            #     3: 16,
            #     # Angle 4 (R Shoulder) -> 'right_shoulder1' (Index 19)
            #     # (We'll use shoulder1 for the main swing)
            #     4: 19,
            #     # Angle 5 (R Elbow) -> 'right_elbow' (Index 22)
            #     5: 22,
            #     # Angle 6 (L Shoulder) -> 'left_shoulder1' (Index 24)
            #     6: 24,
            #     # Angle 7 (L Elbow) -> 'left_elbow' (Index 27)
            #     7: 27,
            #     # Angle 8 (Spine) -> 'abdomen_y' (Index 1)
            #     8: 1
            # }
            joint_mapping = {
                0: 7,   # right_hip_y
                1: 9,   # right_knee
                2: 14,  # left_hip_y
                3: 16,  # left_knee
                4: 19,  # right_shoulder1
                5: 22,  # right_elbow
                6: 24,  # left_shoulder1
                7: 27,  # left_elbow
                8: 1    # abdomen_y
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
        # if len(obs) < 41:
        #     obs = np.pad(obs, (0, 41 - len(obs)), mode='constant')
    
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
        self._disable_default_motors()
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
        self.last_action = action
        scaled_action = action * 150

        if len(action) != len(self.control_joint_indices):
            raise ValueError(
                f"Action length mismatch: expected {len(self.control_joint_indices)}, got {len(action)}"
            )

        # 🔧 Apply torques to only these 9 controlled joints
        pybullet.setJointMotorControlArray(
            bodyUniqueId=self.humanoid_id,
            jointIndices=self.control_joint_indices,
            controlMode=pybullet.TORQUE_CONTROL,
            forces=scaled_action
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


    # def _compute_reward(self, obs):
    #     """
    #     Compute reward based on current state (obs) and last applied torques

    #     Reward formula:
    #         R_t = w_vel * r_vel + w_live * r_live - w_energy * r_energy
    #     """
        
    #     # Extract simulation data
    #     base_lin_vel, base_ang_vel = pybullet.getBaseVelocity(self.humanoid_id,physicsClientId = self.client)
    #     pos, orn = pybullet.getBasePositionAndOrientation(self.humanoid_id, physicsClientId=self.client)
    #     z_height = pos[2]

    #     # Forward velocity reward (positive x direction)
    #     r_vel = max(base_lin_vel[0],0.0) 
    #     # so if move +ve x then the x direction vel is positive x reward but if going in -ve x direction then the vel is -ve so reward is 0

    #     height_threshold = 0.7
    #     r_live = 1.0 if z_height > height_threshold else 0.0
    #     # gives +ve 1 reward if height > threshold else the reward is 0 and this is live reward

    #     r_energy = np.sum(np.square(getattr(self, 'last_action', np.zeros(self.num_actions))))
    #     # here it penalizes for large or abrupt torques for smooth and efficient motiion
    #     # self.last_action has torques applied in prev step
    #     # we do np.square so large torques get more penality than smaller ones
    #     # so getattr ensures that if last_action don't exist then it sets all the action values to 0
        
    #     # defining the weights
    #     w_vel = 1.0 # makes forward walking the main objective 
    #     w_live = 0.2 # gives a small incentive for staying upright
    #     w_energy = 0.001 # small penalty to encourage efficiency

    #     # final reward
    #     reward = (w_vel * r_vel) + (w_live * r_live) - (w_energy * r_energy)

    #     if self._check_termination(obs):
    #         reward -= 5.0

    #     return float(reward)
    def _compute_reward(self, obs):
        """
        Enhanced reward for proper walking behavior
        """
        # Extract data
        base_lin_vel, base_ang_vel = pybullet.getBaseVelocity(
            self.humanoid_id, physicsClientId=self.client
        )
        pos, orn = pybullet.getBasePositionAndOrientation(
            self.humanoid_id, physicsClientId=self.client
        )
        
        x_pos, y_pos, z_height = pos
        roll, pitch, yaw = pybullet.getEulerFromQuaternion(orn)
        
        # Get joint states for foot height detection
        right_knee_state = pybullet.getJointState(self.humanoid_id, 9)  # right_knee
        left_knee_state = pybullet.getJointState(self.humanoid_id, 16)  # left_knee
        
        # ===== REWARD COMPONENTS =====
        
        # 1. Forward velocity reward (main objective)
        forward_vel = base_lin_vel[0]
        r_forward = forward_vel  # Reward forward movement (can be negative)
        
        # 2. Penalize sideways drift
        sideways_vel = abs(base_lin_vel[1])
        r_drift = -sideways_vel
        
        # 3. Upright posture reward (encourage staying vertical)
        # Penalize tilting too much
        r_upright = -abs(roll) - abs(pitch)
        
        # 4. Height maintenance (stay at reasonable height)
        target_height = 1.0  # Adjust based on your humanoid
        height_penalty = -abs(z_height - target_height)
        r_height = height_penalty
        
        # 5. Knee bending reward (encourage leg lifting)
        # Reward when knees are bent (not fully extended)
        right_knee_angle = abs(right_knee_state[0])
        left_knee_angle = abs(left_knee_state[0])
        
        # Encourage some knee bending (walking requires bending knees)
        min_knee_bend = 0.1  # Minimum desired knee angle
        r_knee_right = max(0, right_knee_angle - min_knee_bend)
        r_knee_left = max(0, left_knee_angle - min_knee_bend)
        r_knee_activity = (r_knee_right + r_knee_left) * 0.5
        
        # 6. Alternating gait reward (encourage stepping pattern)
        # When one knee is bent more than the other = better gait
        knee_diff = abs(right_knee_angle - left_knee_angle)
        r_gait = knee_diff * 0.3  # Reward alternating leg movement
        
        # 7. Energy efficiency (but less penalizing than before)
        action = getattr(self, 'last_action', np.zeros(self.num_actions))
        r_energy = -np.sum(np.square(action)) * 0.0005  # Reduced penalty
        
        # 8. Foot contact alternation (if you can access foot link states)
        # This would require detecting which foot is on ground
        # r_foot_contact = ... (advanced, optional)
        
        # 9. Stability penalty (penalize excessive angular velocity)
        ang_vel_magnitude = np.sqrt(sum(v**2 for v in base_ang_vel))
        r_stability = -ang_vel_magnitude * 0.1
        
        # ===== WEIGHT COMBINATION =====
        weights = {
            'forward': 5.0,      # Main objective: move forward
            'drift': 2.0,        # Stay on straight path
            'upright': 1.0,      # Maintain balance
            'height': 1.0,       # Keep reasonable height
            'knee': 1.5,         # Encourage leg movement
            'gait': 1.0,         # Encourage alternating steps
            'energy': 1.0,       # Efficiency (minimal now)
            'stability': 0.5     # Don't spin wildly
        }
        
        reward = (
            weights['forward'] * r_forward +
            weights['drift'] * r_drift +
            weights['upright'] * r_upright +
            weights['height'] * r_height +
            weights['knee'] * r_knee_activity +
            weights['gait'] * r_gait +
            weights['energy'] * r_energy +
            weights['stability'] * r_stability
        )
        
        # Large penalty for falling
        if self._check_termination(obs):
            reward -= 10.0
        
        # Bonus for sustained forward progress
        if forward_vel > 0.5 and z_height > 0.8:
            reward += 2.0
    
        return float(reward)

    def _disable_default_motors(self):
        """Disable built-in PD controllers"""
        num_joints = pybullet.getNumJoints(self.humanoid_id, physicsClientId=self.client)
        for joint_idx in range(num_joints):
            pybullet.setJointMotorControl2(
                bodyUniqueId=self.humanoid_id,
                jointIndex=joint_idx,
                controlMode=pybullet.VELOCITY_CONTROL,
                force=0,
                physicsClientId=self.client
            )

    def _check_termination(self, obs):
        """More nuanced termination"""
        z = obs[2]  # torso height
        
        # Get orientation
        pos, orn = pybullet.getBasePositionAndOrientation(
            self.humanoid_id, physicsClientId=self.client
        )
        roll, pitch, yaw = pybullet.getEulerFromQuaternion(orn)
        
        # Terminate if:
        # 1. Too low (fell down)
        if z < 0.5:
            return True
        
        # 2. Tipped over (extreme tilt)
        if abs(roll) > 0.8 or abs(pitch) > 0.8:  # ~45 degrees
            return True
        
        # 3. Moved too far sideways (off course)
        if abs(pos[1]) > 2.0:
            return True
        
        return False


    def close(self):
        pybullet.disconnect(self.client)
