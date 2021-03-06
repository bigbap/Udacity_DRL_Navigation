import numpy as np
from numpy import random as rnd
from src.sum_tree import SumTree
import torch

DEFAULT_BUFFER_SIZE = 1000
DEFAULT_BATCH_SIZE = 64

PRIORITY_ALPHA = 0.3
EP = 0.01
BETA = 0.4
BETA_INCREMENT = 0.001

class Experience:
    def __init__(self, state, action, reward, state_prime, done, idx):
        self.state = state
        self.action = action
        self.reward = reward
        self.state_prime = state_prime
        self.done = done
        self.idx = idx

class ReplayBuffer():
    def __init__(self, max_length=DEFAULT_BUFFER_SIZE, batch_size=DEFAULT_BATCH_SIZE):
        self.max_length = max_length
        self.batch_size = batch_size

        self.buffer = []
        self.head = -1

    def move_head(self):
        self.head = (self.head + 1) % self.max_length
        
        if len(self.buffer) < self.max_length:
            self.buffer += [None]

    def add(self, state, action, reward, state_prime, done):
        self.move_head()

        self.buffer[self.head] = Experience(state, action, reward, state_prime, done, self.head)
  
    def sample(self):
        batch = rnd.choice(self.buffer, size=self.batch_size)

        return self.prepare_batch(batch)

    def prepare_batch(self, batch, is_w=None):
        if is_w is None:
            is_w = [1.0] * len(batch)

        states = torch.from_numpy(np.vstack([e.state for e in batch if e is not None])).float()
        actions = torch.from_numpy(np.vstack([e.action for e in batch if e is not None])).long()
        rewards = torch.from_numpy(np.vstack([e.reward for e in batch if e is not None])).float()
        next_states = torch.from_numpy(np.vstack([e.state_prime for e in batch if e is not None])).float()
        dones = torch.from_numpy(np.vstack([e.done for e in batch if e is not None]).astype(np.uint8)).float()
        idx_list = torch.from_numpy(np.vstack([e.idx for e in batch if e is not None])).int()
        is_weights = torch.from_numpy(np.vstack(is_w)).float()

        return (states, actions, rewards, next_states, dones, idx_list, is_weights)

    def update_td(self, *args, **kwargs):
        pass

    def __len__(self):
        return len(self.buffer)

class PriorityReplayBuffer(ReplayBuffer):
    def __init__(self, max_length=DEFAULT_BUFFER_SIZE, batch_size=DEFAULT_BATCH_SIZE, alpha=PRIORITY_ALPHA, ep=EP, beta=BETA, beta_increment=BETA_INCREMENT):
        super().__init__(max_length, batch_size)
        
        self.max_priority = 10000.0
        self.beta = beta
        self.beta_increment = beta_increment

        self.get_priority = lambda td: (np.abs(td) + ep) ** alpha

        self.tree = SumTree(max_entries=max_length)

    def add(self, state, action, reward, state_prime, done):
        super().add(state, action, reward, state_prime, done)

        priority = self.get_priority(self.max_priority)
        
        self.tree.update(self.head, priority)

    def sample(self):
        if len(self) < self.batch_size:
            raise "batch size larger than current buffer size"

        segment = self.tree.total() / self.batch_size

        batch = []
        priorities = []
        for i in range(self.batch_size):
            idx, priority = self.tree.sample(rnd.uniform(segment * i, segment * (i + 1)))
            batch.append(self.buffer[idx])
            priorities.append(priority)

        self.beta = np.min([1., self.beta + self.beta_increment])

        sample_probs = priorities / self.tree.total()   # calculate probabilities
        is_w = (len(self) * sample_probs) ** -self.beta # calculate is_weights
        is_w /= np.max(is_w)                            # normalize is_weights

        return self.prepare_batch(batch, is_w=is_w)

    def update_td(self, idx, predictions, targets):
        errors = (targets - predictions).data.numpy()
        priorities = self.get_priority(errors)
        
        for i in range(self.batch_size):
            self.tree.update(idx[i], priorities[i])