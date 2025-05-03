import numpy as np
import pandas as pd
from scipy.stats import entropy

class SBEOS_Environment:
    def __init__(self,max_timesteps=180,reward=20,penalty=5,pressure=0.2,window_size=10,time_dependence=4,noise_mean_min=-0.5,noise_mean_max=0.5,noise_std_min=0.5,noise_std_max=1.0):
        self.max_timesteps = max_timesteps
        self.reward = reward
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
        self.init_band()
    def init_band(self):
        t = int(self.time_dependence)
        assert t>=1,"Time_dependence must be greater than or equal to 1"
        self.band = np.random.choice([0, 1], size=t)
        self.actual_band = self.band.copy()
        t_m = {}
        state_space = [0,1]
        from itertools import product
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

        # Estimate noise as deviation from ideal binary signal
        binary_estimate = np.round(sign_v)  # Expected clean values (0 or 1)
        noise = sign_v - binary_estimate

        estimated_noise_mean = np.mean(noise)
        estimated_noise_std = np.std(noise)

        # Optional denoising (shrink toward rounded binary signal using estimated noise stats)
        # Here we use a basic thresholding scheme to clean noisy signal
        denoised_v = np.clip(sign_v - estimated_noise_mean, 0, 1)
        denoised_v = np.round(denoised_v)  # binary again after correction

        vc = np.bincount(denoised_v.astype(int), minlength=2)
        pdf = vc / len(denoised_v)
        entropy_v = entropy(pdf, base=2) if not np.all(pdf == 0) else 0
        smoothed_change = np.mean(np.abs(np.diff(denoised_v)))
        energy = np.sum(denoised_v ** 2) / len(denoised_v)
        last_state_duration = self.time_since_last_change(denoised_v)

        observation = np.concatenate([
            denoised_v,                      # Denoised window
            [entropy_v,                     # Entropy of the cleaned window
            estimated_noise_mean,         # Estimated noise mean
            estimated_noise_std,          # Estimated noise std
            smoothed_change,              # Average change magnitude
            energy,                        # Signal energy
            last_state_duration]          # Time since last state switch
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
        # self.band = np.array(self.band[-self.window_size:])
        # self.actual_band = np.array(self.actual_band[-self.window_size:])
        self.band = np.array([])
        self.actual_band = np.array([])
        self.init_band()
        self.current_timestep = 0
        self.current_state = self.generate_state()
        return self.generate_observation_state()
    
    def cal_reward(self,actual,prediction):
        if actual == prediction:
            return self.reward
        elif actual != prediction and actual == 1:
            return -self.penalty - self.pressure*self.current_timestep
            # return -self.penalty
        else:
            return self.penalty - self.pressure*(self.current_timestep//2)
            # return self.penalty
        # else:
        #     return -self.penalty
    
    def step(self,action):
        self.current_timestep += 1
        reward = self.cal_reward(int(self.actual_band[-1]),action)
        #print("actual",int(self.actual_band[-1]),"prediction",action)
        self.current_state = self.generate_state()
        observation = self.generate_observation_state()
        done = self.current_timestep >= self.max_timesteps
        info = {
            "timestep": self.current_timestep,
            "correct_prediction": self.actual_current_state == action,
            "state": int(self.actual_band[-2])
        }
        return observation,reward,done,info
        

        
class SBEDS_Environment:
    def __init__(self,max_timesteps=180,reward=10,penalty=5,pressure=0.1,window_size=10):
        self.max_timesteps = max_timesteps
        self.reward = reward
        self.penalty = penalty
        self.pressure = pressure
        self.window_size = window_size
        self.band = np.array([])
        self.actual_band = np.array([])
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
        self.actual_band = np.array([t1, t2])
        self.noise_mean = np.random.uniform(-0.1, 0.1)
        self.noise_std = np.random.uniform(0.01, 0.1)
    def generate_state(self):
        p_2 = tuple(self.band[-2:])
        t_m2 = self.transiton_matrix
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
        # self.band = self.band[-self.window_size:]
        # self.current_timestep = 0
        # self.current_state = self.generate_state()
        # return self.generate_observation_state()
        self.band = np.array([])
        self.actual_band = np.array([])
        self.init_band()
        self.current_timestep = 0
        self.current_state = self.generate_state()
        return self.generate_observation_state()
    
    def cal_reward(self,actual,prediction):
        if actual == prediction:
            return self.reward
        elif actual != prediction and actual == 1:
            return -self.penalty - self.pressure*self.current_timestep
        else:
            return self.penalty - self.pressure*self.current_timestep
    
    def step(self,action):
        self.current_timestep += 1
        reward = self.cal_reward(self.current_state,action)
        self.current_state = self.generate_state()
        observation = self.generate_observation_state()
        done = self.current_timestep >= self.max_timesteps
        info = {
            "timestep": self.current_timestep,
            "correct_prediction": self.actual_current_state == action,
            "state": self.actual_current_state
        }
        return observation,reward,done,info
        

class SBOS_Environment:
    def __init__(self,max_timesteps=180,reward=10,penalty=5,pressure=0.1,window_size=10):
        self.max_timesteps = max_timesteps
        self.reward = reward
        self.penalty = penalty
        self.pressure = pressure
        self.window_size = window_size
        self.band = np.array([])
        self.actual_band = np.array([])
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
        self.actual_band = np.array([t1, t2])
        self.noise_mean = np.random.uniform(-0.1, 0.1)
        self.noise_std = np.random.uniform(0.01, 0.1)
    def generate_state(self):
        p_2 = tuple(self.band[-2:])
        t_m2 = self.transiton_matrix
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
        # self.band = np.array(self.band[-self.window_size:])
        # self.actual_band = np.array(self.actual_band[-self.window_size:])
        self.band = np.array([])
        self.actual_band = np.array([])
        self.init_band()
        self.current_timestep = 0
        self.current_state = self.generate_state()
        return self.current_state
    
    def cal_reward(self,actual,prediction):
        if actual == prediction:
            return self.reward
        elif actual != prediction and actual == 1:
            return -self.penalty - self.pressure*self.current_timestep
        else:
            return self.penalty - self.pressure*self.current_timestep
    
    def step(self,action):
        self.current_timestep += 1
        reward = self.cal_reward(self.current_state,action)
        self.current_state = self.generate_state()
        observation = self.current_state
        done = self.current_timestep >= self.max_timesteps
        info = {
            "timestep": self.current_timestep,
            "correct_prediction": self.actual_current_state == action,
            "state": self.actual_current_state
        }
        return observation,reward,done,info
        

