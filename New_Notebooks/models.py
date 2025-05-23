import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque
from scipy.stats import entropy
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
from environments import *
import torch.nn.functional as F
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(QNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )

    def forward(self, state):
        return self.net(state)

class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        return (
            torch.tensor(state, dtype=torch.float32).to(device).unsqueeze(1),
            torch.tensor(action).to(device),
            torch.tensor(reward).to(device),
            torch.tensor(next_state, dtype=torch.float32).to(device).unsqueeze(1),
            torch.tensor(done).float().to(device)
        )

    def __len__(self):
        return len(self.buffer)
    
class CNNQNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(CNNQNetwork, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(16, 64, kernel_size=3, padding=1)
        self.dropout = nn.Dropout(0.2)
        self.fc1 = nn.Linear(64, 128)
        self.out = nn.Linear(128, output_dim)

    def forward(self, x):
        # x shape: [batch_size, 1] -> reshape to [batch_size, 1, 1]
        x = x.view(-1, 1, 1)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = torch.mean(x, dim=2)  # global average pooling
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        return self.out(x)

# class ANNLSTMNetwork(nn.Module):
#     def __init__(self, input_dim, output_dim, lstm_hidden=64):
#         super(ANNLSTMNetwork, self).__init__()
#         self.fc1 = nn.Linear(input_dim, 64)
#         self.dropout = nn.Dropout(0.2)
#         self.lstm = nn.LSTM(input_size=64, hidden_size=lstm_hidden, batch_first=True)
#         self.out = nn.Linear(lstm_hidden, output_dim)

#     def forward(self, x):
#         if len(x.shape) == 1:  # [features]
#             x = x.unsqueeze(0)  # [1, features]
#         if len(x.shape) == 2:  # [batch, features]
#             x = self.fc1(x)
#             x = self.dropout(x)
#             x = x.unsqueeze(1)  # [batch, seq_len=1, features=32]
#         _, (h_n, _) = self.lstm(x)
#         return self.out(h_n.squeeze(0))
class ANNLSTMNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, lstm_hidden=64):
        super(ANNLSTMNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)  # Project input to 64-dim
        self.dropout = nn.Dropout(0.2)
        self.lstm = nn.LSTM(input_size=64, hidden_size=lstm_hidden, batch_first=True)
        self.out = nn.Linear(lstm_hidden, output_dim)

    def forward(self, x):
        # Ensure input is [batch, features]
        if x.dim() == 1:
            x = x.unsqueeze(0)  # [1, features]
        
        x = self.fc1(x)        # [batch, 64]
        x = self.dropout(x)
        x = x.unsqueeze(1)     # [batch, seq_len=1, 64]
        
        lstm_out, (h_n, _) = self.lstm(x)  # h_n: [1, batch, hidden_size]
        
        # h_n[-1] is the last layer's hidden state for each batch
        return self.out(h_n[-1])  
# class CNNLSTMNetwork(nn.Module):
#     def __init__(self, input_dim, output_dim, lstm_hidden=64):
#         super(CNNLSTMNetwork, self).__init__()
#         self.conv1 = nn.Conv1d(input_dim, 32, kernel_size=3, padding=1)
#         self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
#         self.lstm = nn.LSTM(input_size=64, hidden_size=lstm_hidden, batch_first=True)
#         self.dropout = nn.Dropout(0.3)
#         self.out = nn.Linear(lstm_hidden, output_dim)

#     def forward(self, x):
#         # x: [batch_size, 1] => [batch_size, 1, 1] for Conv1d
#         x = x.view(-1, 1, 1)
#         x = F.relu(self.conv1(x))  # [batch, 16, 1]
#         x = F.relu(self.conv2(x))  # [batch, 32, 1]
#         x = x.permute(0, 2, 1)     # [batch, seq_len=1, features=32]
#         _, (h_n, _) = self.lstm(x)
#         x = self.dropout(h_n.squeeze(0))
#         return self.out(x)

class CNNLSTMNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, lstm_hidden=64):
        super(CNNLSTMNetwork, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.lstm = nn.LSTM(input_size=64, hidden_size=lstm_hidden, batch_first=True)
        self.dropout = nn.Dropout(0.3)
        self.out = nn.Linear(lstm_hidden, output_dim)

    def forward(self, x):
        # x: [batch_size, seq_len] -> [batch_size, 1, seq_len]
        x = x.unsqueeze(1)
        x = F.relu(self.conv1(x))        # [batch, 32, seq_len]
        x = F.relu(self.conv2(x))        # [batch, 64, seq_len]
        x = x.permute(0, 2, 1)           # [batch, seq_len, 64]
        _, (h_n, _) = self.lstm(x)       # h_n: [1, batch, hidden]
        x = self.dropout(h_n.squeeze(0)) # [batch, hidden]
        return self.out(x)               # [batch, output_dim]

class TransformerDQN(nn.Module):
    def __init__(self, input_dim, output_dim, d_model=64, nhead=4, num_layers=2, dim_feedforward=128, dropout=0.1):
        super(TransformerDQN, self).__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(d_model, output_dim)

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)  # [batch, seq_len=1, input_dim]
        x = self.embedding(x)
        x = self.transformer_encoder(x)
        return self.fc_out(x[:, -1, :])  # use the final token representation

class BayesianDQN(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=64, dropout_rate=0.2):
        super(BayesianDQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.out = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)  # remains active in eval mode
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        return self.out(x)

    def enable_dropout(self):
        for m in self.modules():
            if isinstance(m, nn.Dropout):
                m.train()

def get_model(model_type, input_dim, output_dim):
    if model_type == 'cnn':
        return CNNQNetwork(input_dim, output_dim)
    elif model_type == 'ann_lstm':
        return ANNLSTMNetwork(input_dim, output_dim)
    elif model_type == 'cnn_lstm':
        return CNNLSTMNetwork(input_dim, output_dim)
    elif model_type == 'dqn':
        return QNetwork(input_dim, output_dim)
    elif model_type == 'transformer':
        return TransformerDQN(input_dim, output_dim)
    elif model_type == 'bayesian':
        return BayesianDQN(input_dim, output_dim)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")


# def train_dqn(env, model_type='dqn',num_episodes=200, batch_size=64, gamma=0.99,
#               lr=1e-3, epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=0.995,
#               target_update_freq=10, initial_window=10, final_window=2):

#     action_dim = 2
#     state_dim = env.window_size + 5

#     q_net = get_model(model_type, state_dim, action_dim).to(device)
#     target_net = get_model(model_type, state_dim, action_dim).to(device)
#     target_net.load_state_dict(q_net.state_dict())

#     optimizer = optim.Adam(q_net.parameters(), lr=lr)
#     replay_buffer = ReplayBuffer()

#     epsilon = epsilon_start

#     all_rewards = []
#     metrics_per_episode = []
#     epoch_lengths = []
#     window_sizes = []

#     for episode in range(num_episodes):
#         # --- Curriculum window size decay ---
#         # new_window_size = int(max(
#         #     final_window,
#         #     initial_window - ((initial_window - final_window) * episode / num_episodes)
#         # ))
#         # env.window_size = new_window_size
#         # window_sizes.append(new_window_size)
#         window_sizes.append(env.window_size)
#         state = env.reset()
#         done = False
#         total_reward = 0
#         y_true, y_pred, y_scores = [], [], []
#         steps = 0

#         while not done:
#             if random.random() < epsilon:
#                 action = random.choice([0, 1])
#             else:
#                 with torch.no_grad():
#                     q_vals = q_net(torch.tensor([state], dtype=torch.float32).to(device))
#                     action = q_vals.argmax().item()

#             next_state, reward, done, info = env.step(action)
#             replay_buffer.push(state, action, reward, next_state, done)
#             state = next_state
#             total_reward += reward

#             y_true.append(info["state"])
#             y_pred.append(action)
#             with torch.no_grad():
#                 if model_type == 'dqn':
#                     state_tensor = torch.tensor([[state]], dtype=torch.float32).to(device)
#                 else:
#                     state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)  # [1, feature_dim]
#                 q_values = q_net(state_tensor)
#                 probs = torch.softmax(q_values, dim=-1).cpu().numpy().flatten()
#                 y_scores.append(probs[1] if len(probs) == 2 else 0.5)

#             steps += 1

#             if len(replay_buffer) >= batch_size:
#                 s, a, r, ns, d = replay_buffer.sample(batch_size)
#                 # print(f"s.shape: {s.shape}")
#                 # print(f"a.shape: {a.shape}")
#                 # print(f"q_net(s).shape: {q_net(s).shape}")
#                 #q_vals = q_net(s).gather(1, a.unsqueeze(1)).squeeze()
#                 # q_vals = q_net(s).gather(1, a)
#                 if s.dim() == 3:
#                     s = s.view(s.size(0), -1)
#                     ns = ns.view(ns.size(0), -1)

#                 # Ensure action indices are shaped properly
#                 if a.dim() == 1:
#                     a = a.unsqueeze(1)

#                 q_vals = q_net(s).gather(1, a).squeeze(1) 
#                 max_next_q = target_net(ns).max(1)[0]
#                 expected_q = r + gamma * max_next_q * (1 - d)
#                 loss = nn.MSELoss()(q_vals, expected_q.detach())

#                 optimizer.zero_grad()
#                 loss.backward()
#                 optimizer.step()

#         if episode % target_update_freq == 0:
#             target_net.load_state_dict(q_net.state_dict())

#         epsilon = max(epsilon * epsilon_decay, epsilon_end)
#         all_rewards.append(total_reward)
#         epoch_lengths.append(steps)

#         acc = accuracy_score(y_true, y_pred)
#         prec = precision_score(y_true, y_pred, zero_division=0)
#         rec = recall_score(y_true, y_pred, zero_division=0)
#         f1 = f1_score(y_true, y_pred, zero_division=0)
#         try:
#             roc_auc = roc_auc_score(y_true, y_scores,zero_division=0)
#         except:
#             roc_auc = 0.5

#         metrics_per_episode.append((acc, prec, rec, f1, roc_auc))
        
#         if episode % 10 == 0:
#             print(f"Ep {episode} | Reward={total_reward:.1f} | Acc={acc:.2f}, Prec={prec:.2f}, Rec={rec:.2f}, F1={f1:.2f}, ROC-AUC={roc_auc:.2f}")

#     return all_rewards, metrics_per_episode, epoch_lengths, window_sizes, y_true, y_scores

def train_dqn(env, model_type='dqn', num_episodes=200, batch_size=64, gamma=0.99,
              lr=1e-3, epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=0.995,
              target_update_freq=10, initial_window=10, final_window=2):

    action_dim = 2
    state_dim = env.window_size + 5

    q_net = get_model(model_type, state_dim, action_dim).to(device)
    target_net = get_model(model_type, state_dim, action_dim).to(device)
    target_net.load_state_dict(q_net.state_dict())

    optimizer = optim.Adam(q_net.parameters(), lr=lr)
    replay_buffer = ReplayBuffer()

    epsilon = epsilon_start

    all_rewards = []
    metrics_per_episode = []
    epoch_lengths = []
    window_sizes = []

    all_y_true = []
    all_y_scores = []

    for episode in range(num_episodes):
        window_sizes.append(env.window_size)
        state = env.reset()
        done = False
        total_reward = 0
        ep_y_true, ep_y_pred, ep_y_scores = [], [], []
        steps = 0

        while not done:
            if random.random() < epsilon:
                action = random.choice([0, 1])
            else:
                with torch.no_grad():
                    q_vals = q_net(torch.tensor([state], dtype=torch.float32).to(device))
                    action = q_vals.argmax().item()

            next_state, reward, done, info = env.step(action)
            replay_buffer.push(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

            ep_y_true.append(info["state"])
            ep_y_pred.append(action)

            with torch.no_grad():
                if model_type == 'dqn':
                    state_tensor = torch.tensor([[state]], dtype=torch.float32).to(device)
                else:
                    state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
                q_values = q_net(state_tensor)
                probs = torch.softmax(q_values, dim=-1).cpu().numpy().flatten()
                ep_y_scores.append(probs[1] if len(probs) == 2 else 0.5)

            steps += 1

            if len(replay_buffer) >= batch_size:
                s, a, r, ns, d = replay_buffer.sample(batch_size)

                if s.dim() == 3:
                    s = s.view(s.size(0), -1)
                    ns = ns.view(ns.size(0), -1)
                if a.dim() == 1:
                    a = a.unsqueeze(1)

                q_vals = q_net(s).gather(1, a).squeeze(1)
                max_next_q = target_net(ns).max(1)[0]
                expected_q = r + gamma * max_next_q * (1 - d)
                loss = nn.MSELoss()(q_vals, expected_q.detach())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        if episode % target_update_freq == 0:
            target_net.load_state_dict(q_net.state_dict())

        epsilon = max(epsilon * epsilon_decay, epsilon_end)
        all_rewards.append(total_reward)
        epoch_lengths.append(steps)

        acc = accuracy_score(ep_y_true, ep_y_pred)
        prec = precision_score(ep_y_true, ep_y_pred, zero_division=0)
        rec = recall_score(ep_y_true, ep_y_pred, zero_division=0)
        f1 = f1_score(ep_y_true, ep_y_pred, zero_division=0)
        try:
            roc_auc = roc_auc_score(ep_y_true, ep_y_scores)
        except:
            roc_auc = 0.5

        metrics_per_episode.append((acc, prec, rec, f1, roc_auc))
        all_y_true.append(ep_y_true)
        all_y_scores.append(ep_y_scores)

        if episode % 10 == 0:
            print(f"Ep {episode} | Reward={total_reward:.1f} | Acc={acc:.2f}, Prec={prec:.2f}, Rec={rec:.2f}, F1={f1:.2f}, ROC-AUC={roc_auc:.2f}")

    return all_rewards, metrics_per_episode, epoch_lengths, window_sizes, all_y_true, all_y_scores


import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay
from scipy.interpolate import make_interp_spline
import numpy as np

def smooth_curve(x, y, points=300):
    x_new = np.linspace(min(x), max(x), points)
    try:
        spline = make_interp_spline(x, y)
        y_smooth = spline(x_new)
        return x_new, y_smooth
    except Exception as e:
        return x, y 

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay

def average_every_n(x, y, n=10):
    x_avg = []
    y_avg = []
    for i in range(0, len(y), n):
        x_avg.append(np.mean(x[i:i + n]))
        y_avg.append(np.mean(y[i:i + n]))
    return x_avg, y_avg

# def plot_metrics(rewards, metrics, epoch_lengths, window_sizes, y_true, y_scores, avg_window=10):
#     accs, precs, recs, f1s, aucs = zip(*metrics)
#     episodes = list(range(len(rewards)))

#     # --- Rewards ---
#     x_r, y_r = average_every_n(episodes, rewards, avg_window)
#     plt.figure(figsize=(10, 4))
#     plt.plot(x_r, y_r, label='Mean Reward', color='teal')
#     plt.title("Average Cumulative Rewards")
#     plt.xlabel("Episode")
#     plt.ylabel("Reward")
#     plt.grid(True)
#     plt.show()

#     # --- Accuracy ---
#     x_acc, y_acc = average_every_n(episodes, accs, avg_window)
#     plt.figure(figsize=(10, 4))
#     plt.plot(x_acc, y_acc, label='Mean Accuracy', color='blue')
#     plt.title("Accuracy per Episode (Averaged)")
#     plt.xlabel("Episode")
#     plt.ylabel("Accuracy")
#     plt.grid(True)
#     plt.show()

#     # --- Precision ---
#     x_prec, y_prec = average_every_n(episodes, precs, avg_window)
#     plt.figure(figsize=(10, 4))
#     plt.plot(x_prec, y_prec, label='Mean Precision', color='orange')
#     plt.title("Precision per Episode (Averaged)")
#     plt.xlabel("Episode")
#     plt.ylabel("Precision")
#     plt.grid(True)
#     plt.show()

#     # --- Recall ---
#     x_rec, y_rec = average_every_n(episodes, recs, avg_window)
#     plt.figure(figsize=(10, 4))
#     plt.plot(x_rec, y_rec, label='Mean Recall', color='green')
#     plt.title("Recall per Episode (Averaged)")
#     plt.xlabel("Episode")
#     plt.ylabel("Recall")
#     plt.grid(True)
#     plt.show()

#     # --- F1-score ---
#     x_f1, y_f1 = average_every_n(episodes, f1s, avg_window)
#     plt.figure(figsize=(10, 4))
#     plt.plot(x_f1, y_f1, label='Mean F1-Score', color='red')
#     plt.title("F1 Score per Episode (Averaged)")
#     plt.xlabel("Episode")
#     plt.ylabel("F1 Score")
#     plt.grid(True)
#     plt.show()

#     # --- Epoch Lengths ---
#     x_e, y_e = average_every_n(episodes, epoch_lengths, avg_window)
#     # plt.figure(figsize=(10, 4))
#     # plt.plot(x_e, y_e, label='Avg Episode Length', color='purple')
#     # plt.title("Steps per Episode")
#     # plt.xlabel("Episode")
#     # plt.ylabel("Steps")
#     # plt.grid(True)
#     # plt.show()

#     # --- Window Size ---
#     x_w, y_w = average_every_n(episodes, window_sizes, avg_window)
#     # plt.figure(figsize=(10, 4))
#     # plt.plot(x_w, y_w, label='Avg Window Size', color='brown')
#     # plt.title("Window Size over Episodes")
#     # plt.xlabel("Episode")
#     # plt.ylabel("Window Size")
#     # plt.grid(True)
#     # plt.show()

#     # --- ROC Curve ---
#     try:
#         fpr, tpr, _ = roc_curve(y_true, y_scores)
#         roc_auc = roc_auc_score(y_true, y_scores)
#         plt.figure(figsize=(6, 6))
#         plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}", color='darkorange')
#         plt.plot([0, 1], [0, 1], 'k--', alpha=0.6)
#         plt.title("ROC Curve (Last Episode)")
#         plt.xlabel("False Positive Rate")
#         plt.ylabel("True Positive Rate")
#         plt.legend()
#         plt.grid(True)
#         plt.show()
#     except Exception as e:
#         print("ROC Curve could not be generated:", e)

#     # --- Confusion Matrix (last episode) ---
#     try:
#         preds = [1 if s >= 0.5 else 0 for s in y_scores]
#         cm = confusion_matrix(y_true, preds)
#         disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
#         disp.plot(cmap=plt.cm.Blues, values_format='d')
#         plt.title("Confusion Matrix (Last Episode)")
#         plt.grid(False)
#         plt.show()
#     except Exception as e:
#         print("Confusion Matrix could not be generated:", e)

def plot_metrics(rewards, metrics, epoch_lengths, window_sizes, all_y_true, all_y_scores, avg_window=10):
    accs, precs, recs, f1s, aucs = zip(*metrics)
    episodes = list(range(len(rewards)))

    # --- Rewards ---
    x_r, y_r = average_every_n(episodes, rewards, avg_window)
    plt.figure(figsize=(10, 4))
    plt.plot(x_r, y_r, label='Mean Reward', color='teal')
    plt.title("Average Cumulative Rewards")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.grid(True)
    plt.show()

    # --- Accuracy ---
    x_acc, y_acc = average_every_n(episodes, accs, avg_window)
    plt.figure(figsize=(10, 4))
    plt.plot(x_acc, y_acc, label='Mean Accuracy', color='blue')
    plt.title("Accuracy per Episode (Averaged)")
    plt.xlabel("Episode")
    plt.ylabel("Accuracy")
    plt.grid(True)
    plt.show()

    # --- Precision ---
    x_prec, y_prec = average_every_n(episodes, precs, avg_window)
    plt.figure(figsize=(10, 4))
    plt.plot(x_prec, y_prec, label='Mean Precision', color='orange')
    plt.title("Precision per Episode (Averaged)")
    plt.xlabel("Episode")
    plt.ylabel("Precision")
    plt.grid(True)
    plt.show()

    # --- Recall ---
    x_rec, y_rec = average_every_n(episodes, recs, avg_window)
    plt.figure(figsize=(10, 4))
    plt.plot(x_rec, y_rec, label='Mean Recall', color='green')
    plt.title("Recall per Episode (Averaged)")
    plt.xlabel("Episode")
    plt.ylabel("Recall")
    plt.grid(True)
    plt.show()

    # --- F1-score ---
    x_f1, y_f1 = average_every_n(episodes, f1s, avg_window)
    plt.figure(figsize=(10, 4))
    plt.plot(x_f1, y_f1, label='Mean F1-Score', color='red')
    plt.title("F1 Score per Episode (Averaged)")
    plt.xlabel("Episode")
    plt.ylabel("F1 Score")
    plt.grid(True)
    plt.show()

    # --- ROC Curve (All Episodes Combined) ---
    try:
        y_true_flat = np.concatenate(all_y_true)
        y_scores_flat = np.concatenate(all_y_scores)

        fpr, tpr, _ = roc_curve(y_true_flat, y_scores_flat)
        roc_auc = roc_auc_score(y_true_flat, y_scores_flat)

        plt.figure(figsize=(6, 6))
        plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}", color='darkorange')
        plt.plot([0, 1], [0, 1], 'k--', alpha=0.6)
        plt.title("ROC Curve (All Episodes)")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend()
        plt.grid(True)
        plt.show()
    except Exception as e:
        print("ROC Curve could not be generated:", e)

    # --- Confusion Matrix (All Episodes Combined) ---
    try:
        preds = [1 if s >= 0.5 else 0 for s in y_scores_flat]
        cm = confusion_matrix(y_true_flat, preds)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
        disp.plot(cmap=plt.cm.Blues, values_format='d')
        plt.title("Confusion Matrix (All Episodes)")
        plt.grid(False)
        plt.show()
    except Exception as e:
        print("Confusion Matrix could not be generated:", e)


def plot_reward_comparison(reward_dict, avg_window=10, title="Reward Comparison Across Models"):
    """
    reward_dict: Dict[str, List[float]]
        e.g., {"ANN_LSTM": [...], "CNN_LSTM": [...], "Conv1D": [...]}
    avg_window: int
        Number of episodes to average over
    """

    def average_every_n(x, y, n=10):
        x_avg, y_avg = [], []
        for i in range(0, len(y), n):
            x_avg.append(np.mean(x[i:i+n]))
            y_avg.append(np.mean(y[i:i+n]))
        return x_avg, y_avg

    plt.figure(figsize=(12, 6))

    for label, rewards in reward_dict.items():
        episodes = list(range(len(rewards)))
        x_smooth, y_smooth = average_every_n(episodes, rewards, avg_window)
        plt.plot(x_smooth, y_smooth, label=label, linewidth=2)

    plt.title(title)
    plt.xlabel("Episode")
    plt.ylabel("Average Reward")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


import optuna

def objective(trial):
    # Sample environment hyperparameters
    reward = trial.suggest_float("reward", 5, 50)
    penalty = trial.suggest_float("penalty", 5.0, 20)
    pressure = trial.suggest_float("pressure", 0.1, 10.0)

    # Create environment with trial-specific parameters
    env = SBEOS_Environment(reward=reward, penalty=penalty, pressure=pressure)

    # Train the DQN with the trial environment
    all_rewards, metrics_per_episode, _, _, y_true, y_scores = train_dqn(
        env=env,
        model_type='dqn',
        num_episodes=50,  # Shorter run for fast tuning; increase later
        batch_size=64,
        gamma=0.99,
        lr=1e-3,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay=0.99,
        target_update_freq=10
    )

    # --- Objective Metrics ---
    mean_reward = np.mean(all_rewards[-10:])
    mean_accuracy = np.mean([m[0] for m in metrics_per_episode[-10:]])


    # Combined metric (weighted sum): Feel free to change weights
    return mean_reward, mean_accuracy