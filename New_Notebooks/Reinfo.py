import numpy as np
from scipy.stats import entropy

class ReinforcementEnvironment:
    def __init__(self, num_bands, energy_cost=2, reward_factor=5, weight=5, max_timestep=180):
        self.num_bands = num_bands
        self.energy_cost = energy_cost
        self.reward_factor = reward_factor
        self.max_timestep = max_timestep
        self.weight = weight
        self.signal_band = {band: [] for band in range(self.num_bands)}
        self.current_timestep = 0
        self.transition_matrixes = {band: {} for band in range(self.num_bands)}
        self.init_bands()
        self.current_state = self.get_current_state()
    
    def init_bands(self):
        """Initialize each band with two initial signal values (0 or 1)"""
        for band in range(self.num_bands):
            # First signal chosen with equal probability
            t1 = np.random.choice([0, 1])
            
            # Second signal chosen with random probability distribution
            t_m1 = np.random.rand(2,2)
            t_m1 /= t_m1.sum(axis=1,keepdims=True)  # Normalize to create valid probability distribution
            t2 = np.random.choice([0, 1], p=t_m1[t1])
            # t_m2 = {
            #     (0, 0): np.random.rand(2),
            #     (0, 1): np.random.rand(2),
            #     (1, 0): np.random.rand(2),
            #     (1, 1): np.random.rand(2)
            # }
            # for k in t_m2:
            #     t_m2[k] /= t_m2[k].sum()
            t_m2 = {
            (0, 0): np.random.dirichlet([1, 1]),  # Generates a valid probability distribution over {0,1}
            (0, 1): np.random.dirichlet([1, 1]),
            (1, 0): np.random.dirichlet([1, 1]),
            (1, 1): np.random.dirichlet([1, 1])
            }
            self.transition_matrixes[band] = t_m2
            self.signal_band[band] = [t1, t2]  
    
    def step(self, action):
        """
        Execute one time step within the environment
        
        Args:
            action: tuple (band, prediction) where band is the selected frequency band
                   and prediction is the predicted signal value (0 or 1)
        
        Returns:
            tuple: (observation, reward, done, info)
        """
        self.current_timestep += 1
        
        band = action[0]
        prediction = action[1]
        
        reward = self._calculate_reward(self.current_state[band], prediction)
        
        self.generate_state()
        
        observation = self.construct_observation_space()
        
        done = self.current_timestep >= self.max_timestep
        
        info = {
            "timestep": self.current_timestep,
            "correct_prediction": self.current_state[band] == prediction,
            "state": self.current_state
        }
        
        return observation, reward, done, info
    
    def _calculate_reward(self, actual_signal, prediction):
        """Calculate reward based on prediction accuracy and signal value"""
        if actual_signal == prediction:
            # Correct prediction
            reward = self.reward_factor * self.weight - self.energy_cost
        elif actual_signal == 0:
            # Incorrect prediction when signal is 0
            reward = self.reward_factor - self.energy_cost
        else:  # actual_signal == 1
            # Incorrect prediction when signal is 1
            reward = self.reward_factor - self.energy_cost * self.weight
        
        return reward
    
    def generate_state(self):
        """Generate next state for all bands based on transition probabilities"""
        for band in range(self.num_bands):
            # Get last two signals for this band
            p_2 = tuple(self.signal_band[band][-2:])
            
            t_m2 = self.transition_matrixes[band]
            
            next_signal = np.random.choice([0, 1], p=t_m2[p_2])
            
            self.signal_band[band].append(next_signal)
            self.signal_band[band].pop(0)
        
        # Update current state
        self.current_state = self.get_current_state()
        
        return self.current_state
    
    def get_current_state(self):
        """Return the current state as a list of the most recent signal for each band"""
        return [self.signal_band[band][-1] for band in range(self.num_bands)]
    
    def reset(self):
        """Reset the environment to initial state and return initial observation"""
        self.signal_band = {band: [] for band in range(self.num_bands)}
        self.current_timestep = 0
        self.init_bands()
        self.current_state = self.get_current_state()
        return self.construct_observation_space()
    
    def construct_observation_space(self, window_size=10):
        """
        Construct observation space with entropy calculations for each band
        
        Args:
            window_size: Number of recent signals to consider for entropy calculation
            
        Returns:
            list: Entropy values for each band
        """
        observation = []
        for band in range(self.num_bands):
            signal_values = np.array(self.signal_band[band][-window_size:])
            
            if len(signal_values) <= window_size:
                entropy_value = 0
            else:
                value_counts = np.bincount(signal_values, minlength=2)
                
                probability_distribution = value_counts / len(signal_values)
                
                # Handle edge cases
                if np.all(probability_distribution == 0):
                    entropy_value = 0
                else:
                    # Calculate entropy using scipy function
                    entropy_value = entropy(probability_distribution, base=2)
            
            observation.append(entropy_value)
        
        return observation
    
    def soft_reset(self):
        self.signal_band = {band: self.signal_band[band][-2:] for band in range(self.num_bands)}
        self.current_timestep = 0
        self.generate_state()
        return self.construct_observation_space()
        