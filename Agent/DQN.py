import torch
from torch import nn
import random

class Agent:
    def __init__(self, action_dim, obs_dim, eps, eps_low, eps_decay):
        self.action_dim = action_dim
        self.obs_dim = obs_dim
        self.eps = eps
        self.eps_low = eps_low
        self.eps_decay = eps_decay

    def eps_scheduler(self):
        self.eps *= self.eps_decay
        self.eps = max(self.eps, self.eps_low)


class DQNAgent(Agent):
    def __init__(self, action_dim, obs_dim, batch_size=128, gamma=0.99, eps=1, eps_low=0.1, eps_decay=0.999, tau=0.05):
        super().__init__(action_dim, obs_dim, eps, eps_low, eps_decay)
        self.policy_network = DQN(action_dim, obs_dim)
        self.target_network = DQN(action_dim, obs_dim)
        self.target_network.load_state_dict(self.policy_network.state_dict())

        self.batch_size = batch_size
        self.tau = tau
        self.gamma = gamma

        self.loss_fn = nn.MSELoss()
        self.optim = torch.optim.Adam(self.policy_network.parameters(), lr=1e-3)

    def step(self, obs):
        with torch.no_grad():
            obs = torch.unsqueeze(torch.from_numpy(obs), 0)
            action = torch.argmax(self.target_network(obs)).item()
            return random.choice(range(self.action_dim)) if random.random() < self.eps else action

    def learn(self, buffer):
        obs, rewards, actions, next_obs, terminated = buffer.sample(self.batch_size)
        obs = torch.Tensor(obs)
        rewards = torch.Tensor(rewards)
        actions = torch.Tensor(actions).long()
        next_obs = torch.Tensor(next_obs)
        terminated = torch.Tensor(terminated).long()


        Q_policy = self.policy_network(obs).gather(1, actions.reshape(-1, 1))
        Q_target = torch.max(self.target_network(next_obs), dim=1)[0]

        y = rewards + self.gamma * Q_target * (1 - terminated)

        loss = self.loss_fn(Q_policy.reshape(-1), y.reshape(-1))
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()


    def soft_update(self):
        for target_param, policy_param in zip(self.target_network.parameters(), self.policy_network.parameters()):
            target_param.data.copy_(self.tau * policy_param.data + (1.0 - self.tau ) * target_param.data)


class DQN(nn.Module):
    def __init__(self, action_dim, obs_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )

    def forward(self, x):
        return self.model(x)