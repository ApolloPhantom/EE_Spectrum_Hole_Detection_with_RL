import numpy as np
import pandas as pd
from scipy.stats import entropy
from itertools import product
class SBEOS_Environment:
    def __init__(self,max_timesteps=180,energy_cost=10,reward=10,penalty=5,pressure=0.1,window_size=10,time_dependence=4,
                 noise_mean_min=-0.9,noise_mean_max=0.9,noise_std_min=0.2,noise_std_max=0.6):
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
    # def generate_observation_state(self):
    #     sign_v = np.array(self.band[-self.window_size:])

    #     if len(sign_v) < self.window_size:
    #         pad_len = self.window_size - len(sign_v)
    #         sign_v = np.concatenate((np.zeros(pad_len), sign_v))

    #     binary_estimate = np.round(sign_v)  
    #     noise = sign_v - binary_estimate

    #     estimated_noise_mean = np.mean(noise)
    #     estimated_noise_std = np.std(noise)
    #     denoised_v = np.clip(sign_v - estimated_noise_mean, 0, 1)
    #     denoised_v = np.round(denoised_v)  

    #     vc = np.bincount(denoised_v.astype(int), minlength=2)
    #     pdf = vc / len(denoised_v)
    #     entropy_v = entropy(pdf, base=2) if not np.all(pdf == 0) else 0
    #     smoothed_change = np.mean(np.abs(np.diff(denoised_v)))
    #     energy = np.sum(denoised_v ** 2) / len(denoised_v)
    #     last_state_duration = self.time_since_last_change(denoised_v)

    #     observation = np.concatenate([
    #         denoised_v,                      # Denoised window
    #         # sign_v,
    #         [entropy_v,                     # Entropy of the cleaned window
    #         estimated_noise_mean,         # Estimated noise mean
    #         estimated_noise_std,          # Estimated noise std
    #         smoothed_change,              # Average change magnitude
    #         energy,                        # Signal energy
    #         last_state_duration]          # Time since last state switch
    #     ])

    #     return observation
    # def time_since_last_change(self, sign_v):
    #     if len(sign_v) < 2:
    #         return 0
    #     rev = sign_v[::-1]
    #     for i in range(1, len(rev)):
    #         if rev[i] != rev[0]:
    #             return i
    #     return len(rev)
    
    def generate_observation_state(self):
        sign_v = np.array(self.band[-self.window_size:])

        if len(sign_v) < self.window_size:
            pad_len = self.window_size - len(sign_v)
            sign_v = np.concatenate((np.zeros(pad_len), sign_v))

        # Use sign_v directly (no denoising)
        binary_v = np.round(sign_v).astype(int)  # used for idle/busy estimation and other binary metrics

        # Calculate idle and busy fractions
        vc = np.bincount(binary_v, minlength=2)
        total = len(binary_v)
        idle_fraction = vc[0] / total if total > 0 else 0
        busy_fraction = vc[1] / total if total > 0 else 0

        # Probability distribution for entropy
        pdf = vc / total if total > 0 else np.zeros(2)
        entropy_v = entropy(pdf, base=2) if not np.all(pdf == 0) else 0

        # Temporal features from binary_v
        smoothed_change = np.mean(np.abs(np.diff(binary_v)))
        energy = np.sum(binary_v ** 2) / total  # effectively same as busy_fraction
        last_state_duration = self.time_since_last_change(binary_v)

        # Compose final observation
        observation = np.concatenate([
            sign_v,                      # Raw signal window (unchanged)
            [entropy_v,                  # Entropy
            idle_fraction,             # Idle fraction
            busy_fraction,             # Busy fraction
            smoothed_change,           # Mean abs diff
            energy,                    # Energy of binary signal
            last_state_duration]       # Time since last state switch
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
