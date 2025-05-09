import numpy as np
import pandas as pd
from scipy.stats import entropy
from itertools import product

import torch
import torch.nn as nn

class CompactDAE(nn.Module):
    def __init__(self, window_size):
        super(CompactDAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(window_size, window_size // 2),
            nn.ReLU(),
            nn.Linear(window_size // 2, window_size // 4),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(window_size // 4, window_size // 2),
            nn.ReLU(),
            nn.Linear(window_size // 2, window_size),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


class SBEOS_Environment:
    def __init__(self,max_timesteps=180,energy_cost=10,reward=10,penalty=5,pressure=0.1,window_size=10,time_dependence=4,
                 noise_mean_min=-0.1,noise_mean_max=0.1,noise_std_min=0.2,noise_std_max=0.6):
        self.max_timesteps = max_timesteps
        self.reward = reward
        self.min_energy_cost = energy_cost
        self.max_energy_cost = energy_cost*pressure
        self.penalty = penalty
        self.pressure = pressure
        self.window_size = window_size
        self.time_dependence = time_dependence
        self.noise_mean_min = noise_mean_min
        self.noise_mean_max = noise_mean_max
        self.noise_std_min = noise_std_min
        self.noise_std_max = noise_std_max
        self.band = np.array([]) #Storage for Noisy States
        self.actual_band = np.array([]) #Storage for Actual States
        self.energy_spent = np.array([]) #Storage of energy spent
        self.true_state = []
        self.predictions = []
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dae = CompactDAE(window_size).to(self.device)
        self.dae.load_state_dict(torch.load("dae_compact_trained.pth"))
        self.dae.eval()
        self.init_band()
    def init_band(self):
        t = int(self.time_dependence)
        assert t>=1,"Time_dependence must be greater than or equal to 1"
        self.band = np.random.choice([0, 1], size=t)
        self.actual_band = self.band.copy()
        t_m = {}
        state_space = [0,1]
        for past in product(state_space, repeat=t):
            # Introduce a bias factor
            bias_factor = np.random.uniform(0.1, 0.9)
            # Base transition probabilities biased toward 0 or 1
            transition_probs = [bias_factor, 1 - bias_factor]
            # Add randomness (noise) to make transitions less deterministic
            transition_probs = np.array(transition_probs) + np.random.normal(0, 0.1, 2)
            transition_probs = np.clip(transition_probs, 0, 1)
            transition_probs /= transition_probs.sum()  # Normalize to sum to 1

            t_m[past] = transition_probs

        self.transition_matrix = t_m
        self.noise_mean = np.random.uniform(self.noise_mean_min, self.noise_mean_max)
        self.noise_std = np.random.uniform(self.noise_std_min, self.noise_std_max)
    def generate_state(self):
        p_2 = tuple(self.band[-self.time_dependence:])
        t_m2 = self.transition_matrix
        next_state = np.random.choice([0,1],p=t_m2[p_2])
        self.actual_band = np.append(self.actual_band, next_state)
        noise = np.random.normal(self.noise_mean, self.noise_std)
        noisy_state = np.round(np.clip(next_state + noise, 0, 1))
        noisy_state = int(noisy_state)
        self.band = np.append(self.band,noisy_state)
        self.actual_current_state = self.actual_band[-1]
        return noisy_state
    def generate_observation_state(self):
        sign_v = np.array(self.band[-self.window_size:])

        if len(sign_v) < self.window_size:
            pad_len = self.window_size - len(sign_v)
            sign_v = np.concatenate((np.zeros(pad_len), sign_v))

        # Normalize input for the DAE
        input_tensor = torch.tensor(sign_v, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            denoised_v = self.dae(input_tensor).cpu().squeeze(0).numpy()

        denoised_v = np.clip(denoised_v, 0, 1)
        denoised_v = np.round(denoised_v)

        # Feature engineering
        vc = np.bincount(denoised_v.astype(int), minlength=2)
        pdf = vc / len(denoised_v)
        entropy_v = entropy(pdf, base=2) if not np.all(pdf == 0) else 0
        smoothed_change = np.mean(np.abs(np.diff(denoised_v)))
        energy = np.sum(denoised_v ** 2) / len(denoised_v)
        last_state_duration = self.time_since_last_change(denoised_v)

        observation = np.concatenate([
            denoised_v,
            [entropy_v,
            sign_v.mean() - denoised_v.mean(),  # rough est. noise mean
            np.std(sign_v - denoised_v),       # rough est. noise std
            smoothed_change,
            energy,
            last_state_duration]
        ])
        
        return observation

    def time_since_last_change(self, sign_v):
        if len(sign_v) < 2:
            return 0
        rev = sign_v[::-1]
        for i in range(1, len(rev)):
            if rev[i] != rev[0]:
                return i
        return len(rev)
    def reset(self):
        self.band = np.array([])
        self.actual_band = np.array([])
        self.init_band()
        self.current_timestep = 0
        self.current_state = self.generate_state()
        return self.generate_observation_state()
    
    def cal_reward(self,actual,action):
        if action == 0 and actual == 0:
            return self.reward 
        elif action == 0 and actual == 1:
            return -self.penalty
        elif action == 1 and actual == 1:
            return self.reward 
        elif action == 1 and actual == 0:
            return -self.penalty 
    
    def step(self,action):
        reward = self.cal_reward(int(self.actual_band[-1]),action%2)
        #print("actual",int(self.actual_band[-1]),"prediction",action)
        self.current_state = self.generate_state()
        observation = self.generate_observation_state()
        done = self.current_timestep >= self.max_timesteps
        info = {
            "timestep": self.current_timestep,
            "state": int(self.actual_band[-2]),
            "energy_cost": self.min_energy_cost if action < 2 else self.max_energy_cost
        }
        self.current_timestep += 1
        return observation,reward,done,info
        
