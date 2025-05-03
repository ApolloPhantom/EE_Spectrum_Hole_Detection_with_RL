import numpy as np
import pandas as pd
from scipy.stats import entropy

import gym
from gym import spaces
import numpy as np
from scipy.stats import entropy

# class SBEOS_Environment(gym.Env):
#     metadata = {"render_modes": ["human"], "render_fps": 4}

#     def __init__(self, max_timesteps=180, reward=10, penalty=5, pressure=0.1, window_size=10):
#         super(SBEOS_Environment, self).__init__()
#         self.max_timesteps = max_timesteps
#         self.reward = reward
#         self.penalty = penalty
#         self.pressure = pressure
#         self.window_size = window_size
#         self.band = np.array([])
#         self.actual_band = np.array([])
#         self.current_timestep = 0
#         self.observation_dim = self.window_size + 6  # window + entropy + idle_frac + busy_frac + smoothed_change + energy + duration
#         self.action_space = spaces.Discrete(2)  # 0 or 1
#         self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(self.observation_dim,), dtype=np.float32)
#         self.seed()
#         self.init_band()

#     def seed(self, seed=None):
#         np.random.seed(seed)

#     def init_band(self):
#         t1 = np.random.choice([0, 1])
#         t2 = np.random.choice([0, 1])
#         t3 = np.random.choice([0, 1])
#         t4 = np.random.choice([0, 1])
#         t_m4 = {}
#         for a in [0, 1]:
#             for b in [0, 1]:
#                 for c in [0, 1]:
#                     for d in [0, 1]:
#                         bias_factor = np.random.uniform(0.1, 0.9)
#                         transition_probs = [bias_factor, 1 - bias_factor]  
#                         transition_probs = np.array(transition_probs) + np.random.normal(0, 0.1, 2)
#                         transition_probs = np.clip(transition_probs, 0, 1)
#                         transition_probs /= transition_probs.sum()
#                         t_m4[(a, b, c, d)] = transition_probs
#         self.transition_matrix = t_m4
#         self.band = np.array([t1, t2, t3, t4])
#         self.actual_band = np.array([t1, t2, t3, t4])
#         self.noise_mean = np.random.uniform(-0.5, 0.5)
#         self.noise_std = np.random.uniform(0.5, 1.0)

#     def generate_state(self):
#         p_2 = tuple(self.band[-4:])
#         next_state = np.random.choice([0, 1], p=self.transition_matrix[p_2])
#         self.actual_band = np.append(self.actual_band, next_state)
#         noise = np.random.normal(self.noise_mean, self.noise_std)
#         noisy_state = np.round(np.clip(next_state + noise, 0, 1)).astype(int)
#         self.band = np.append(self.band, noisy_state)
#         self.actual_current_state = self.actual_band[-1]
#         return noisy_state

#     def generate_observation_state(self):
#         sign_v = np.array(self.band[-self.window_size:])
#         if len(sign_v) < self.window_size:
#             pad_len = self.window_size - len(sign_v)
#             sign_v = np.concatenate((np.zeros(pad_len), sign_v))
#         vc = np.bincount(sign_v.astype(int), minlength=2)
#         pdf = vc / len(sign_v)
#         entropy_v = entropy(pdf, base=2) if not np.all(pdf == 0) else 0
#         idle_frac = pdf[0]
#         busy_frac = pdf[1]
#         smoothed_change = np.mean(np.abs(np.diff(sign_v)))
#         energy = np.sum(sign_v ** 2) / len(sign_v)
#         last_state_duration = self.time_since_last_change(sign_v)
#         observation = np.concatenate([
#             sign_v,
#             [entropy_v, idle_frac, busy_frac, smoothed_change, energy, last_state_duration]
#         ])
#         return observation.astype(np.float32)

#     def time_since_last_change(self, sign_v):
#         if len(sign_v) < 2:
#             return 0
#         rev = sign_v[::-1]
#         for i in range(1, len(rev)):
#             if rev[i] != rev[0]:
#                 return i
#         return len(rev)

#     def reset(self, seed=None, options=None):
#         super().reset(seed=seed)
#         self.seed(seed)
#         self.band = np.array([])
#         self.actual_band = np.array([])
#         self.init_band()
#         self.current_timestep = 0
#         self.current_state = self.generate_state()
#         observation = self.generate_observation_state()
#         info = {}
#         return observation, info

#     def cal_reward(self, actual, prediction):
#         if actual == prediction:
#             return self.reward
#         elif actual != prediction and actual == 1:
#             return -self.penalty - self.pressure * self.current_timestep
#         else:
#             return self.penalty - self.pressure * self.current_timestep

#     def step(self, action):
#         self.current_timestep += 1
#         reward = self.cal_reward(int(self.current_state), action)
#         self.current_state = self.generate_state()
#         observation = self.generate_observation_state()
#         done = self.current_timestep >= self.max_timesteps
#         info = {
#             "timestep": self.current_timestep,
#             "correct_prediction": self.actual_current_state == action,
#             "state": int(self.actual_band[-2])
#         }
#         return observation, reward, done, False, info  # (obs, reward, terminated, truncated, info)

#     def render(self, mode="human"):
#         print(f"Timestep: {self.current_timestep} | Current: {self.current_state} | True: {self.actual_current_state}")

#     def close(self):
#         pass

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from scipy.stats import entropy

class SBEOS_Environment(gym.Env):
    def __init__(self, max_timesteps=180, reward=10, penalty=5, pressure=0.1, window_size=10):
        super(SBEOS_Environment, self).__init__()
        self.max_timesteps = max_timesteps
        self.reward_value = reward
        self.penalty = penalty
        self.pressure = pressure
        self.window_size = window_size
        self.observation_size = self.window_size + 6
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.observation_size,), dtype=np.float32)

        self.band = np.array([])
        self.actual_band = np.array([])
        self.init_band()
        self.current_timestep = 0
        self.current_state = self.generate_state()

    def init_band(self):
        t1, t2, t3, t4 = np.random.choice([0, 1], 4)
        self.transition_matrix = {
            (a, b, c, d): self._normalized_transition_probs()
            for a in [0, 1] for b in [0, 1] for c in [0, 1] for d in [0, 1]
        }
        self.band = np.array([t1, t2, t3, t4])
        self.actual_band = np.array([t1, t2, t3, t4])
        self.noise_mean = np.random.uniform(-0.5, 0.5)
        self.noise_std = np.random.uniform(0.5, 1.0)

    def _normalized_transition_probs(self):
        bias = np.random.uniform(0.1, 0.9)
        probs = np.array([bias, 1 - bias]) + np.random.normal(0, 0.1, 2)
        probs = np.clip(probs, 0, 1)
        return probs / probs.sum()

    def generate_state(self):
        key = tuple(self.band[-4:])
        next_state = np.random.choice([0, 1], p=self.transition_matrix[key])
        self.actual_band = np.append(self.actual_band, next_state)
        noise = np.random.normal(self.noise_mean, self.noise_std)
        noisy_state = np.round(np.clip(next_state + noise, 0, 1)).astype(int)
        self.band = np.append(self.band, noisy_state)
        self.actual_current_state = self.actual_band[-1]
        return noisy_state

    def generate_observation_state(self):
        sign_v = np.array(self.band[-self.window_size:])
        if len(sign_v) < self.window_size:
            sign_v = np.concatenate((np.zeros(self.window_size - len(sign_v)), sign_v))
        vc = np.bincount(sign_v.astype(int), minlength=2)
        pdf = vc / len(sign_v)
        entropy_v = entropy(pdf, base=2) if not np.all(pdf == 0) else 0
        idle_frac, busy_frac = pdf[0], pdf[1]
        smoothed_change = np.mean(np.abs(np.diff(sign_v)))
        energy = np.sum(sign_v ** 2) / len(sign_v)
        last_state_duration = self.time_since_last_change(sign_v)
        return np.concatenate([
            sign_v,
            [entropy_v, idle_frac, busy_frac, smoothed_change, energy, last_state_duration]
        ]).astype(np.float32)

    def time_since_last_change(self, sign_v):
        if len(sign_v) < 2:
            return 0
        rev = sign_v[::-1]
        for i in range(1, len(rev)):
            if rev[i] != rev[0]:
                return i
        return len(rev)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.band = np.array([])
        self.actual_band = np.array([])
        self.init_band()
        self.current_timestep = 0
        self.current_state = self.generate_state()
        observation = self.generate_observation_state()
        return observation, {}

    def cal_reward(self, actual, prediction):
        if actual == prediction:
            return self.reward_value
        elif actual == 1:
            return -self.penalty - self.pressure * self.current_timestep
        else:
            return self.penalty - self.pressure * self.current_timestep

    def step(self, action):
        reward = self.cal_reward(int(self.current_state), action)
        self.current_state = self.generate_state()
        observation = self.generate_observation_state()
        self.current_timestep += 1
        terminated = False
        truncated = self.current_timestep >= self.max_timesteps
        info = {
            "timestep": self.current_timestep,
            "noisy": int(self.band[-2]),
            "state": int(self.actual_band[-2])
        }
        return observation, reward, terminated, truncated, info
