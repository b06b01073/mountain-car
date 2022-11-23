import collections
import random
import torch

class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = collections.deque(maxlen=capacity)
        self.size = 0

    def insert(self, experience):
        self.buffer.append(experience)

        if self.size < len(self.buffer):
            self.size += 1

    def sample(self, batch_size):
        if batch_size > self.size:
            return None

        batch = random.sample(self.buffer, batch_size)
        obs = []
        rewards = []
        actions = []
        next_obs = []
        terminated = []

        for experience in batch:
            obs.append(experience[0])
            rewards.append(experience[1])
            actions.append(experience[2])
            next_obs.append(experience[3])
            terminated.append(experience[4])


        # process batch before return

        return obs, rewards, actions, next_obs, terminated


    def __len__(self):
        return len(self.buffer)

    