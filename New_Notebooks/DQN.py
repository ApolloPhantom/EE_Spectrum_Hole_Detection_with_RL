import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from collections import deque
import matplotlib.pyplot as plt
from Reinfo import ReinforcementEnvironment

class AdvancedDeepQNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(AdvancedDeepQNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc_out = nn.Linear(128, output_dim)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = x.float()
        if x.dim() == 1:
            x = x.unsqueeze(0)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        return self.fc_out(x)



class ImprovedDQNAgent_Soft:
    def __init__(self, env, input_dim, output_dim):
        self.env = env
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.q_network = AdvancedDeepQNetwork(input_dim, output_dim).to(self.device)
        self.target_network = AdvancedDeepQNetwork(input_dim, output_dim).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.learning_rate = 0.0005
        self.gamma = 0.99
        self.temperature = 1.0
        self.min_temperature = 0.1
        self.temperature_decay = 0.9995
        self.temperature_decay_steps = 500
        self.replay_memory = deque(maxlen=20000)
        self.batch_size = 128
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.learning_rate, weight_decay=1e-5)
        self.loss_fn = F.smooth_l1_loss

    def select_action(self, state):
        self.temperature = max(self.min_temperature, self.temperature * (self.temperature_decay ** (1 / self.temperature_decay_steps)))
        with torch.no_grad():
            state_tensor = torch.tensor(state, dtype=torch.float32).to(self.device).unsqueeze(0)
            q_values = self.q_network(state_tensor).squeeze(0).cpu().numpy()
            scaled_q_values = q_values / self.temperature - np.max(q_values / self.temperature)
            exp_q = np.exp(scaled_q_values)
            action_probs = exp_q / np.sum(exp_q)
            action_idx = np.random.choice(len(q_values), p=action_probs)
            band = action_idx // 2
            prediction = action_idx % 2
            return (band, prediction)

    def store_transition(self, state, action, reward, next_state, done):
        self.replay_memory.append((state, action, reward, next_state, done))

    def experience_replay(self):
        if len(self.replay_memory) < self.batch_size:
            return
        batch = random.sample(self.replay_memory, self.batch_size)
        states = torch.tensor(np.array([b[0] for b in batch]), dtype=torch.float32).to(self.device)
        actions = [b[1] for b in batch]
        rewards = torch.tensor(np.array([b[2] for b in batch]), dtype=torch.float32).to(self.device)
        next_states = torch.tensor(np.array([b[3] for b in batch]), dtype=torch.float32).to(self.device)
        dones = torch.tensor(np.array([b[4] for b in batch]), dtype=torch.float32).to(self.device)
        current_q_values = self.q_network(states)
        next_q_values_main = self.q_network(next_states)
        next_q_values_target = self.target_network(next_states)
        max_next_actions = next_q_values_main.argmax(1)
        max_next_q_values = next_q_values_target.gather(1, max_next_actions.unsqueeze(1)).squeeze(1)
        target_q_values = rewards + (1 - dones) * self.gamma * max_next_q_values
        batch_actions = [action[0] * 2 + action[1] for action in actions]
        current_q_values = current_q_values.gather(1, torch.tensor(batch_actions, dtype=torch.long).unsqueeze(1).to(self.device)).squeeze(1)
        loss = self.loss_fn(current_q_values, target_q_values.detach())
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1)
        self.optimizer.step()

    def train(self, episodes=1000):
        total_rewards = []
        self.env.reset()
        for episode in range(episodes):
            state = self.env.soft_reset()
            total_reward = 0
            done = False
            correct_predictions = 0
            total_predictions = 0

            while not done:
                action = self.select_action(state)
                next_state, reward, done, info = self.env.step(action)

                total_predictions += 1
                if info['correct_prediction']:
                    correct_predictions += 1

                self.store_transition(state, action, reward, next_state, done)
                state = next_state
                total_reward += reward

                self.experience_replay()

            if episode % 100 == 0:
                self.target_network.load_state_dict(self.q_network.state_dict())

            total_rewards.append(total_reward)

            if episode % 50 == 0:
                accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
                print(f"Episode {episode}, "
                      f"Total Reward: {total_reward:.2f}, "
                      f"Accuracy: {accuracy:.2%}, "
                      f"Temperature: {self.temperature:.4f}")
                      
        return total_rewards
    





class ImprovedDQNAgent:
    def __init__(self, env, input_dim, output_dim):
        """
        Enhanced Deep Q-Learning Agent with advanced techniques
        """
        self.env = env
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Advanced network with twin networks
        self.q_network = AdvancedDeepQNetwork(input_dim, output_dim).to(self.device)
        self.target_network = AdvancedDeepQNetwork(input_dim, output_dim).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # More sophisticated hyperparameters
        self.learning_rate = 0.0005  # Reduced learning rate
        self.gamma = 0.99  # Discount factor
        
        # Improved exploration strategy
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.9975  # Slower decay
        self.epsilon_decay_steps = 500  # Decay over 500 episodes
        
        # Prioritized Experience Replay
        self.replay_memory = []
        self.memory_size = 20000
        self.batch_size = 128  # Increased batch size
        
        # Optimizer with adaptive learning rate
        self.optimizer = optim.Adam(
            self.q_network.parameters(), 
            lr=self.learning_rate, 
            weight_decay=1e-5  # L2 regularization
        )
        
        # Loss function with huber loss for more robust learning
        self.loss_fn = F.smooth_l1_loss
    
    # def select_action(self, state):
    #     """
    #     Advanced action selection with adaptive exploration
    #     """
    #     # Adaptive epsilon decay
    #     self.epsilon = max(
    #         self.epsilon_min, 
    #         self.epsilon * (self.epsilon_decay ** (1 / self.epsilon_decay_steps))
    #     )
        
    #     if np.random.rand() <= self.epsilon:
    #         # Exploration with increased randomness in early stages
    #         band = np.random.randint(0, len(state))
    #         prediction = np.random.randint(0, 2)
    #         return (band, prediction)
    #     else:
    #         # Exploitation
    #         with torch.no_grad():
    #             # Convert state to tensor and move to device
    #             state_tensor = torch.tensor(state, dtype=torch.float32).to(self.device)
                
    #             # Ensure correct dimensions
    #             if state_tensor.dim() == 1:
    #                 state_tensor = state_tensor.unsqueeze(0)
                
    #             # Get Q-values
    #             q_values = self.q_network(state_tensor)
                
    #             # Select action
    #             action_idx = torch.argmax(q_values).item()
                
    #             # Convert action index to (band, prediction)
    #             band = action_idx // 2
    #             prediction = action_idx % 2
    #             return (band, prediction)
    # def select_action(self, state):
    #     """
    #     Improved action selection strategy:
    #     - Uses entropy-based exploration
    #     - Implements Boltzmann exploration for balanced action selection
    #     - Adapts epsilon decay for efficient learning
    #     """
    #     self.epsilon = max(
    #         self.epsilon_min, 
    #         self.epsilon * (self.epsilon_decay ** (1 / self.epsilon_decay_steps))
    #     )
        
    #     # Calculate entropy of each band from the state (higher entropy => more exploration)
    #     entropy_values = np.array(state)
        
    #     # Normalize entropy values to use as probabilities
    #     if np.sum(entropy_values) > 0:
    #         entropy_probs = entropy_values / np.sum(entropy_values)
    #     else:
    #         entropy_probs = np.ones_like(entropy_values) / len(entropy_values)

    #     if np.random.rand() <= self.epsilon:
    #         # Exploration: Prefer bands with higher entropy
    #         band = np.random.choice(len(state), p=entropy_probs)
    #         prediction = np.random.randint(0, 2)  # Random action (0 or 1)
    #         return (band, prediction)
    #     else:
    #         # Exploitation using Boltzmann exploration
    #         with torch.no_grad():
    #             state_tensor = torch.tensor(state, dtype=torch.float32).to(self.device).unsqueeze(0)
    #             q_values = self.q_network(state_tensor).squeeze(0).cpu().numpy()
                
    #             # Convert Q-values into probabilities using Boltzmann distribution
    #             temperature = max(0.1, self.epsilon)  # Avoid division by zero
    #             exp_q = np.exp(q_values / temperature)
    #             action_probs = exp_q / np.sum(exp_q)

    #             # Select action based on probability distribution
    #             action_idx = np.random.choice(len(q_values), p=action_probs)
                
    #             # Convert index to (band, prediction)
    #             band = action_idx // 2
    #             prediction = action_idx % 2
                
    #             return (band, prediction)
    def select_action(self, state):
        self.epsilon = max(
            self.epsilon_min, 
            self.epsilon * (self.epsilon_decay ** (1 / self.epsilon_decay_steps))
        )
        
        entropy_values = np.array(state)
        
        if np.sum(entropy_values) > 0:
            entropy_probs = entropy_values / np.sum(entropy_values)
        else:
            entropy_probs = np.ones_like(entropy_values) / len(entropy_values)

        if np.random.rand() <= self.epsilon:
            band = np.random.choice(len(state), p=entropy_probs)
            prediction = np.random.randint(0, 2)
            return (band, prediction)
        else:
            with torch.no_grad():
                state_tensor = torch.tensor(state, dtype=torch.float32).to(self.device).unsqueeze(0)
                q_values = self.q_network(state_tensor).squeeze(0).cpu().numpy()
                
                # Stabilized softmax using max Q-value normalization
                temperature = max(0.5, self.epsilon)  # Ensure temperature is not too low
                q_values -= np.max(q_values)  # Prevent large exponent values
                exp_q = np.exp(q_values / temperature)
                
                sum_exp_q = np.sum(exp_q)
                if sum_exp_q == 0 or np.isnan(sum_exp_q):
                    action_probs = np.ones_like(exp_q) / len(exp_q)  # Default uniform distribution
                else:
                    action_probs = exp_q / sum_exp_q
                
                action_idx = np.random.choice(len(q_values), p=action_probs)
                
                band = action_idx // 2
                prediction = action_idx % 2
                return (band, prediction)

    
    def store_transition(self, state, action, reward, next_state, done):
        """
        Store transition with basic tracking
        """
        # Store transition
        experience = (state, action, reward, next_state, done)
        
        if len(self.replay_memory) < self.memory_size:
            self.replay_memory.append(experience)
        else:
            # Replace random experience if memory is full
            idx = random.randint(0, len(self.replay_memory) - 1)
            self.replay_memory[idx] = experience
    
    def experience_replay(self):
        """
        Enhanced experience replay
        """
        if len(self.replay_memory) < self.batch_size:
            return
        
        # Sample batch
        batch = random.sample(self.replay_memory, self.batch_size)
        
        # Prepare batch tensors using numpy conversion
        states = torch.tensor(np.array([b[0] for b in batch]), dtype=torch.float32).to(self.device)
        actions = [b[1] for b in batch]
        rewards = torch.tensor(np.array([b[2] for b in batch]), dtype=torch.float32).to(self.device)
        next_states = torch.tensor(np.array([b[3] for b in batch]), dtype=torch.float32).to(self.device)
        dones = torch.tensor(np.array([b[4] for b in batch]), dtype=torch.float32).to(self.device)
        
        current_q_values = self.q_network(states)
        
        # Compute target Q values with double DQN
        next_q_values_main = self.q_network(next_states)
        next_q_values_target = self.target_network(next_states)
        
        # Double DQN: select actions from main network, evaluate from target
        max_next_actions = next_q_values_main.argmax(1)
        max_next_q_values = next_q_values_target.gather(1, max_next_actions.unsqueeze(1)).squeeze(1)
        
        target_q_values = rewards + (1 - dones) * self.gamma * max_next_q_values
        
        batch_actions = [action[0] * 2 + action[1] for action in actions]
        current_q_values = current_q_values.gather(1, torch.tensor(batch_actions, dtype=torch.long).unsqueeze(1).to(self.device)).squeeze(1)
        
        loss = self.loss_fn(current_q_values, target_q_values.detach())
        
        self.optimizer.zero_grad()
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1)
        
        self.optimizer.step()
    
    def train(self, episodes=1000):
        """
        Training loop with comprehensive metrics
        """
        total_rewards = []
        self.env.reset()
        for episode in range(episodes):
            # Reset environment
            #state = self.env.reset()
            state = self.env.soft_reset()
            total_reward = 0
            done = False
            correct_predictions = 0
            total_predictions = 0
            
            while not done:
                action = self.select_action(state)
                next_state, reward, done, info = self.env.step(action)
                
                total_predictions += 1
                if info['correct_prediction']:
                    correct_predictions += 1
                
                self.store_transition(state, action, reward, next_state, done)
                
                state = next_state
                total_reward += reward
                
                self.experience_replay()
            
            if episode % 100 == 0:
                self.target_network.load_state_dict(self.q_network.state_dict())
            
            total_rewards.append(total_reward)
            
            if episode % 50 == 0:
                accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
                print(f"Episode {episode}, "
                      f"Total Reward: {total_reward:.2f}, "
                      f"Accuracy: {accuracy:.2%}, "
                      f"Epsilon: {self.epsilon:.4f}")
        
        return total_rewards






def plot_rewards(rewards, window_size=10):
        plt.figure(figsize=(12, 6), facecolor='white')
        plt.plot(rewards, alpha=0.5, color='lightblue', label='Episode Reward')
        moving_average = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
        plt.plot(np.arange(window_size-1, len(rewards)), moving_average, color='blue', linewidth=2, label=f'{window_size}-Episode Moving Avg')
        plt.title('Training Reward over Episodes', fontweight='bold')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show()   




def main():
    np.random.seed(42)
    torch.manual_seed(42)
    random.seed(42)
    num_bands = 10
    env = ReinforcementEnvironment(num_bands)
    input_dim = num_bands
    output_dim = num_bands * 2
    
    soft_agent = ImprovedDQNAgent_Soft(env, input_dim, output_dim)
    soft_rewards = soft_agent.train(episodes=500)
    plot_rewards(soft_rewards)

    epsilon_agent= ImprovedDQNAgent(env, input_dim, output_dim)
    epsilon_rewards= epsilon_agent.train(episodes=500)
    plot_rewards(epsilon_rewards)


if __name__ == "__main__":
    main()
