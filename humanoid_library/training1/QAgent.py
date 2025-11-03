import os
import random
import time
from collections import deque, namedtuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from humanoid_library.simulation.humanoid_env import HumanoidWalkEnv
from humanoid_library.training1.QNet_architecture import QNetwork

#create the tuple that will be used to store in the circular buffer
#it is tuple named transition and stores (state, action index, reward, next state, done which says if the episode ended or not)
Transition = namedtuple('Transition',('state','action_idxs','reward','next_state','done'))

class ReplayBuffer:
    def __init__(self,capacity,device):
        #buffer defined using a deque with the capacity
        self.buffer = deque(maxlen=capacity)
        self.device = device

    def push(self,*args):
        #the *args collects the arguments need to create the transition buffer and unpacks them to individual entities
        self.buffer.append(Transition(*args))

    def sample(self,batch_size):
        batch = random.sample(self.buffer,batch_size)
        batch = Transition(*zip(*batch))
        # Convert to tensors
        states = torch.tensor(np.vstack(batch.state), dtype=torch.float32, device=self.device)
        next_states = torch.tensor(np.vstack(batch.next_state), dtype=torch.float32, device=self.device)
        rewards = torch.tensor(batch.reward, dtype=torch.float32, device=self.device).unsqueeze(1)
        dones = torch.tensor(batch.done, dtype=torch.float32, device=self.device).unsqueeze(1)
        # action_idxs: (batch,n_joints) integers
        action_idxs = torch.tensor(np.vstack(batch.action_idxs), dtype=torch.long, device=self.device)
        return states, action_idxs, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)
    
class DQNAgent:
    def __init__(self,state_dim,n_joints,n_bins,device='cpu',lr=1e-4,gamma=0.99,batch_size=64,buffer_size=200000,min_replay_size=1000,target_update_freq=1000,tau=1.0):
        self.device = torch.device(device)
        self.n_joints = n_joints
        self.n_bins = n_bins
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.target_update_freq = target_update_freq
        self.update_count = 0

        #policy net is the main target network that is being trained
        self.policy_net = QNetwork(state_dim,n_joints,n_bins).to(self.device)
        #a copy used to compute target Q-values
        self.target_net = QNetwork(state_dim,n_joints,n_bins).to(self.device)
        #initially both are the exactly the same
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(),lr=lr)
        self.replay = ReplayBuffer(buffer_size,device=self.device)
        self.min_replay_size = min_replay_size # max no of samples before training starts
        self.loss_fn = nn.MSELoss()

        # torque bins for mapping discrete bins -> continuous torques
        self.torque_bins = self.policy_net.torque_bins.to(self.device)

    def select_action(self,state_np,epsilon):
        """
        Epsilon-greedy action selection
        state_np: numpy arrary shape(state_dim)
        returns:
            action_idxs: np.array shape(n_joints,) of chosen bin indices
            action_torques: np.array shape(n_joints,) continuous torques in [-1,1]
        """

        if random.random() < epsilon:
            action_idxs = np.random.randint(0,self.n_bins,size=(self.n_joints,))
            action_torques = self.torque_bins[action_idxs].cpu().numpy()
            return action_idxs,action_torques
        else:
            #greedy: pick argmax per-joint from policy network
            self.policy_net.eval()
            with torch.no_grad():
                s=torch.tensor(state_np,dtype=torch.float32,device=self.device).unsqueeze(0) #[1,state_dim]
                qvals = self.policy_net(s) # [1,n_joints,n_bins]
                best_idxs = qvals.argmax(dim=2).squeeze(0).cpu().numpy() #[n_joints]
                action_torques = self.torque_bins[best_idxs].cpu().numpy()
            self.policy_net.train() # swtich back to train mode from eval mode
            return best_idxs,action_torques
    
    def push_transition(self,state,action_idxs,reward,next_state,done):
        #store np arrays/lists in buffer
        self.replay.push(np.array(state,dtype=np.float32),np.array(action_idxs,dtype=np.int64),float(reward),np.array(next_state,dtype=np.float32),bool(done))

    def train_step(self):
        if len(self.replay)<self.min_replay_size:
            return None
        
        # self.replay is our experience replay buffer
        # we don't train untill we have at least min_replay_size transitions to sample from
        # prevents learning from tiny or biased data at the very start

        states, action_idxs, rewards, next_states , dones = self.replay.sample(self.batch_size)
        B = states.size(0)
        #states: [B,state_dim]
        #action_idxs: [B,n_joints]
        #rewards: [B,1]
        #dones: [B,1]

        #Current Q-values for chosen action indices (sum across joints -> scalar Q per sample)
        q_vals = self.policy_net(states) #[B,n_joints,n_bins]
        # gather per joint Qs using action_idxs
        chosen_qs = q_vals.gather(2, action_idxs.unsqueeze(-1)).squeeze(-1)  # [B, n_joints]
        current_q = chosen_qs.sum(dim=1, keepdim=True)  # [B,1] scalar Q for each sample

        # Compute next Q (target) using target_net: max over bins per joint, then sum across joints
        with torch.no_grad():
            next_qvals = self.target_net(next_states) #[B,n_joints,n_bins]
            max_next_per_joint = next_qvals.max(dim=2).values #[B,n_joints]
            max_next_q = max_next_per_joint.sum(dim=1,keepdim=True) #[B,1]
            target_q = rewards + (1.0 - dones) * (self.gamma * max_next_q) # (1-dones) ensures if episode done then the reward added is 0
            # the above target_q is Bellman equation
        
        loss = self.loss_fn(current_q,target_q)

        self.optimizer.zero_grad()
        loss.backward()

        #gradient clipping to prevent gradient to shoot up
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(),max_norm=10.0)
        self.optimizer.step()

        #target network update (hard or soft)
        self.update_count += 1
        if self.tau >= 1.0:
            # hard update every target_update_freq copy weights for every target_update_freq steps
            if self.update_count % self.target_update_freq == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())
        else:
            # soft update - slowly move target network to policy network
            for target_param, param in zip(self.target_net.parameters(), self.policy_net.parameters()):
                target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

        return loss.item()

        
    def save(self,path):
        torch.save(self.policy_net.state_dict(),path)

    def load(self,path):
        self.policy_net.load_state_dict(torch.load(path,map_location=self.device))
        self.target_net.load_state_dict(self.policy_net.state_dict())


# Training loops starts here

def train(
        env,
        agent: DQNAgent,
        initial_pose_library,
        episodes = 2000,
        max_steps_per_episode =1000,
        start_epsilon = 1.0,
        end_epsilon = 0.05,
        epsilon_decay_steps=100000,
        log_interval=10,
        model_save_path='dqn_policy.pth'
):
    total_steps = 0
    epsilon = start_epsilon
    epsilon_decay = (start_epsilon - end_epsilon)/max(1,epsilon_decay_steps)
    rewards_history = []

    for ep in range(1,episodes + 1):
        #sample a random initial pose from the library
        init_pose = None
        if initial_pose_library is not None and len(initial_pose_library) > 0:
            idx = random.randrange(len(initial_pose_library))
            init_pose = initial_pose_library[idx].astype(np.float32)
        obs,info = env.reset(initial_pose = init_pose)

        ep_reward = 0.0
        done = False
        truncated = False

        for step in range(max_steps_per_episode):
            total_steps += 1
            #decay epsilon
            if epsilon > end_epsilon:
                epsilon = max(end_epsilon,epsilon - epsilon_decay)
            #select action(action_idxs = bins per joint, action_torques = continuous)
            action_idxs, action_torques = agent.select_action(obs,epsilon)
            #apply to env (env expects full action vector length self.num_actions)
            #if your env expects a larger action vector (e.g., 17), map/apply only first n_joints entries
            # Here ensure action_torques is the same length as env.num_actions or pad with zeros
            env_action = np.zeros(env.num_actions,dtype = np.float32)
            #first n_joints map;
            n_to_apply = min(agent.n_joints, env.num_actions)
            env_action[:n_to_apply] = action_torques[:n_to_apply]

            next_obs, reward, terminated, truncated, info = env.step(env_action)
            done_flag = bool(terminated or truncated)

            agent.push_transition(obs, action_idxs, reward, next_obs, done_flag)
            loss = agent.train_step()

            obs = next_obs
            ep_reward += reward

            if done_flag:
                break

        rewards_history.append(ep_reward)

        # logging
        if ep % log_interval == 0 or ep == 1:
            avg_rew = np.mean(rewards_history[-log_interval:]) if len(rewards_history) >= log_interval else np.mean(rewards_history)
            print(f"Episode {ep:4d} | Steps {total_steps:8d} | EPS {epsilon:.3f} | EpReward {ep_reward:.3f} | Avg{log_interval} {avg_rew:.3f} | Buffer {len(agent.replay)}")
            # save model periodically
            agent.save(model_save_path)

    # final save
    agent.save(model_save_path)
    return rewards_history

if __name__=="__main__":
    #Basic hyperparams
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    state_dim = 41  # your env.obs_dim
    n_joints = 9
    n_bins = 5

    env = HumanoidWalkEnv(render_mode =None)
    agent = DQNAgent(state_dim=state_dim, n_joints=n_joints, n_bins=n_bins, device=device,
                     lr=1e-4, gamma=0.99, batch_size=128, buffer_size=200000,
                     min_replay_size=2000, target_update_freq=1000, tau=1.0)
    
    #load or synthesize an initial pose library
    #to load it
    if os.path.exists('initial_pose_library.npy'):
        initial_pose_library = np.load('initial_pose_library.npy')
        print(f"Loaded initial pose library: {initial_pose_library.shape}")
    else:
        # If don't have poses, create a synthetic library of small perturbations around zeros.
        L = 200  # library size
        initial_pose_library = (np.random.randn(L, n_joints) * 0.1).astype(np.float32)
        np.save('initial_pose_library.npy', initial_pose_library)
        print(f"Created synthetic initial pose library: {initial_pose_library.shape}")

    #training begin
    start_time = time.time()
    rewards = train(env, agent, initial_pose_library,
                    episodes=2000, max_steps_per_episode=env.max_steps,
                    start_epsilon=1.0, end_epsilon=0.05, epsilon_decay_steps=200000,
                    log_interval=10,
                    model_save_path='dqn_policy.pth')
    print("Training finished in", time.time() - start_time)