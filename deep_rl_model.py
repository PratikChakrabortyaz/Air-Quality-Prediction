import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
from sklearn.metrics import r2_score, mean_squared_error
import time


class PPO(nn.Module):
    def __init__(self, input_dim, hidden_dim=192):
        super(PPO, self).__init__()
        self.hidden_dim = hidden_dim
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.actor = nn.Linear(hidden_dim, 1)
        self.critic = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = torch.relu(self.bn1(self.fc1(x).view(-1, self.hidden_dim)))
        x = torch.relu(self.bn2(self.fc2(x).view(-1, self.hidden_dim)))
        return self.actor(x).squeeze(-1), self.critic(x).mean()


class AQIEnv(gym.Env):
    def __init__(self, data, seq_length):
        super(AQIEnv, self).__init__()
        self.data = data
        self.seq_length = seq_length
        self.current_idx = 0
        self.max_steps = len(data) - seq_length

        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(seq_length, data.shape[1] - 1), dtype=np.float32)
        self.action_space = gym.spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)

    def reset(self):
        self.current_idx = 0
        return self._get_observation()

    def step(self, action):
        predicted_aqi = float(np.array(action).flatten()[0])
        actual_aqi = self.data[self.current_idx + self.seq_length, -1]

        rmse = np.abs(predicted_aqi - actual_aqi)
        reward = max(1 - rmse, 0) - 0.1 * np.var(action)

        self.current_idx += 1
        done = self.current_idx >= self.max_steps
        obs = self._get_observation() if not done else np.zeros_like(self._get_observation())

        return obs, reward, done, {}

    def _get_observation(self):
        return self.data[self.current_idx: self.current_idx + self.seq_length, :-1]


def train_ppo(env, model, optimizer, num_epochs=5, gamma=0.99, eps_clip=0.1):
    model.train()
    for epoch in range(num_epochs):
        obs = env.reset()
        total_reward = 0

        for step in range(env.max_steps):
            obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to('cuda')

            action_mean, state_value = model(obs_tensor)
            action_dist = Normal(action_mean, 0.2)
            action = action_dist.sample()

            obs, reward, done, _ = env.step(action.cpu().detach().numpy())
            total_reward += reward

            advantage = (reward - state_value).unsqueeze(-1)
            critic_loss = advantage.pow(2).mean()
            actor_loss = (-action_dist.log_prob(action) * advantage).mean()

            loss = critic_loss + actor_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if done:
                break

        print(f"Epoch {epoch+1}/{num_epochs} | Total Reward: {total_reward:.4f}")


def evaluate_ppo(env, model):
    model.eval()
    obs = env.reset()
    actuals, predictions = [], []

    with torch.no_grad():
        for step in range(env.max_steps):
            obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to('cuda')
            action_mean, _ = model(obs_tensor)
            predicted_aqi = action_mean.cpu().detach().numpy().squeeze()

            actual_aqi = env.data[env.current_idx + env.seq_length, -1]
            predictions.append(predicted_aqi)
            actuals.append(actual_aqi)

            obs, _, done, _ = env.step([predicted_aqi])
            if done:
                break

    test_r2 = r2_score(actuals, predictions)
    test_rmse = mean_squared_error(actuals, predictions, squared=False)

    print(f"Final Model | Test R²: {test_r2:.4f} | Test RMSE: {test_rmse:.4f}")
