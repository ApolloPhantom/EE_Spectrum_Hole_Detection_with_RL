import numpy as np
import pandas as pd
from scipy.stats import entropy

class SBEOS_Environment:
    def __init__(self,max_timesteps=180,reward=10,penalty=5,window_size=10):
        self.max_timesteps = max_timesteps
        self.reward = reward
        self.penalty = penalty
        self.window_size = window_size
        self.band = np.array([])
        self.init_band()
    def init_band(self):
        t1 = np.random.choice([0, 1])
        t_m1 = np.random.rand(2,2)
        t_m1 /= t_m1.sum(axis=1,keepdims=True)
        t2 = np.random.choice([0, 1], p=t_m1[t1])
        t_m2 = {
        (0, 0): np.random.dirichlet([1, 1]),  
        (0, 1): np.random.dirichlet([1, 1]),
        (1, 0): np.random.dirichlet([1, 1]),
        (1, 1): np.random.dirichlet([1, 1])
        }
        self.transiton_matrix = t_m2
        self.band = np.array([t1, t2])
        self.noise_mean = np.random.uniform(-0.1, 0.1)
        self.noise_std = np.random.uniform(0.01, 0.1)
    def generate_state(self):
        p_2 = tuple(self.band[-2:])
        t_m2 = self.transiton_matrix
        next_state = np.random.choice([0,1],p=t_m2[p_2])
        noise = np.random.normal(self.noise_mean, self.noise_std)
        noisy_state = np.round(np.clip(next_state + noise, 0, 1))
        noisy_state = int(noisy_state)
        self.band = np.append(self.band,noisy_state)
        return noisy_state
    def generate_observation_state(self):
        sign_v = np.array(self.band[-self.window_size:])
        if len(sign_v) < self.window_size:
            entropy_v = 0
        else:
            vc = np.bincount(sign_v,minlength=2)
            pdf = vc/len(sign_v)
            if np.all(pdf == 0):
                entropy_v = 0
            else:
                entropy_v = entropy(pdf,base=2)
        return entropy_v
    def reset(self):
        self.band = np.array(self.band[-self.window_size:])
        self.current_timestep = 0
        self.current_state = self.generate_state()
        return self.generate_observation_state()
    
    def cal_reward(self,actual,prediction):
        if actual == prediction:
            return self.reward
        elif actual != prediction and actual == 1:
            return -self.penalty
        else:
            return self.penalty
    
    def step(self,action):
        self.current_timestep += 1
        reward = self.cal_reward(self.current_state,action)
        self.current_state = self.generate_state()
        observation = self.generate_observation_state()
        done = self.current_timestep >= self.max_timesteps
        info = {
            "timestep": self.current_timestep,
            "correct_prediction": self.current_state == action,
            "state": self.current_state
        }
        return observation,reward,done,info
        

        
class SBEDS_Environment:
    def __init__(self,max_timesteps=180,reward=10,penalty=5,window_size=10):
        self.max_timesteps = max_timesteps
        self.reward = reward
        self.penalty = penalty
        self.window_size = window_size
        self.band = np.array([])
        self.observation = np.array([])
        self.init_band()
    def init_band(self):
        t1 = np.random.choice([0, 1])
        t_m1 = np.random.rand(2,2)
        t_m1 /= t_m1.sum(axis=1,keepdims=True)
        t2 = np.random.choice([0, 1], p=t_m1[t1])
        t_m2 = {
        (0, 0): np.random.dirichlet([1, 1]),  
        (0, 1): np.random.dirichlet([1, 1]),
        (1, 0): np.random.dirichlet([1, 1]),
        (1, 1): np.random.dirichlet([1, 1])
        }
        self.transiton_matrix = t_m2
        self.band = np.array([t1, t2])
        self.noise_mean = np.random.uniform(-0.1, 0.1)
        self.noise_std = np.random.uniform(0.01, 0.1)
    def generate_state(self):
        p_2 = tuple(self.band[-2:])
        t_m2 = self.transiton_matrix
        next_state = np.random.choice([0,1],p=t_m2[p_2])
        noise = np.random.normal(self.noise_mean, self.noise_std)
        noisy_state = np.round(np.clip(next_state + noise, 0, 1))
        noisy_state = int(noisy_state)
        self.band = np.append(self.band,noisy_state)
        return noisy_state
    def generate_observation_state(self):
        sign_v = np.array(self.band[-self.window_size:])
        if len(sign_v) < self.window_size:
            entropy_v = 0
        else:
            vc = np.bincount(sign_v,minlength=2)
            pdf = vc/len(sign_v)
            if np.all(pdf == 0):
                entropy_v = 0
            else:
                entropy_v = entropy(pdf,base=2)
        self.observation = np.append(self.observation, entropy_v)
        if len(self.observation) == 1:
            return entropy_v  
    
        return self.observation[-2] - self.observation[-1]
    
    def reset(self):
        self.band = self.band[-self.window_size:]
        self.current_timestep = 0
        self.current_state = self.generate_state()
        return self.generate_observation_state()
    
    def cal_reward(self,actual,prediction):
        if actual == prediction:
            return self.reward
        elif actual != prediction and actual == 1:
            return -self.penalty
        else:
            return self.penalty
    
    def step(self,action):
        self.current_timestep += 1
        reward = self.cal_reward(self.current_state,action)
        self.current_state = self.generate_state()
        observation = self.generate_observation_state()
        done = self.current_timestep >= self.max_timesteps
        info = {
            "timestep": self.current_timestep,
            "correct_prediction": self.current_state == action,
            "state": self.current_state
        }
        return observation,reward,done,info
        

