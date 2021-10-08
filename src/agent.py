import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from src.replay_buffer import ReplayBuffer, PriorityReplayBuffer
from src.model import Model

EP_START = 1
EP_DECAY = 0.99
EP_MIN = 0.01
LR = 0.01
GAMMA = 0.9
TAU = 1e-3

LEARN_EVERY = 4

DEFAULT_BUFFER_SIZE = 1000
DEFAULT_BATCH_SIZE = 64

def DQN_algo(**kwargs):
    gamma = kwargs["gamma"]
    dqn_target = kwargs["dqn_target"]
    return lambda r, s_, d: r + (gamma * dqn_target(s_).detach().max(1)[0].unsqueeze(1) * (1 - d))

def DoubleDQN_algo(**kwargs):
    gamma = kwargs["gamma"]
    dqn_target = kwargs["dqn_target"]
    dqn_local = kwargs["dqn_local"]
    def targets(r, s_, d):
        inner_target_actions = dqn_local(s_).detach().argmax(dim=1).unsqueeze(1)
        return r + (gamma * dqn_target(s_).gather(1, inner_target_actions) * (1 - d))
    return targets


class Agent():
    def __init__(self, state_space_n, action_space_n, learn_every=LEARN_EVERY, gamma=GAMMA, lr=LR,
                 ep_decay=EP_DECAY, ep_start=EP_START, ep_min=EP_MIN, buffer_size=DEFAULT_BUFFER_SIZE, 
                 batch_size=DEFAULT_BATCH_SIZE, model=None, seed=None, targets_algo=DQN_algo, priority_alpha=None):

        self.state_space_n = state_space_n
        self.action_space_n = action_space_n

        self.dqn_local = Model(state_space_n, action_space_n, seed=seed)
        self.dqn_target = Model(state_space_n, action_space_n, seed=seed)
        
        if model != None:
            self.dqn_local.load_state_dict(torch.load(model))
            self.dqn_target.load_state_dict(torch.load(model))
        
        self.ep_start = ep_start
        self.ep_decay = ep_decay
        self.ep_min = ep_min
        self.gamma = gamma
        self.learn_every = learn_every

        self.loss_fn = F.mse_loss
        self.optimizer = optim.Adam(self.dqn_local.parameters(), lr)

        self.buffer_size = buffer_size
        self.batch_size = batch_size

        if priority_alpha != None:
            self.replay_buffer = PriorityReplayBuffer(buffer_size, batch_size, alpha=priority_alpha)
        else:
            self.replay_buffer = ReplayBuffer(buffer_size, batch_size)

        self.t_step = 1

        self.targets_aglo = targets_algo(gamma=gamma, dqn_local=self.dqn_local, dqn_target=self.dqn_target)

    def act(self, state, ep=None):
        state = torch.from_numpy(state).float().unsqueeze(0)

        ep = ep if ep != None else max(self.ep_start * (self.ep_decay ** self.t_step), self.ep_min)

        if np.random.rand() > ep:
            self.dqn_local.eval()
            with torch.no_grad():
                action_values = self.dqn_local(state).data.numpy()
            self.dqn_local.train()

            return np.argmax(action_values)
        else:
            return np.random.choice(range(self.action_space_n))

    def step(self, state, action, reward, state_prime, done):
        self.replay_buffer.add(state, action, reward, state_prime, done)

        if len(self.replay_buffer) >= self.batch_size and self.t_step % self.learn_every == 0:
            self._learn()

            # update target network
            self._soft_update_dqn_target(self.dqn_local, self.dqn_target, TAU)

        self.t_step += 1

    def _learn(self):
        states, actions, rewards, state_primes, dones, idx_list, is_weights = self.replay_buffer.sample()

        predictions = self.dqn_local(states).gather(1, actions)
        targets = self.targets_aglo(rewards, state_primes, dones)

        # update priorities, only applies if PER
        self.replay_buffer.update_td(idx_list, predictions, targets)

        self.optimizer.zero_grad()
        loss = (torch.FloatTensor(is_weights) * self.loss_fn(predictions, targets)).mean()
        loss.backward()
        self.optimizer.step()

    def _soft_update_dqn_target(self, dqn_local, dqn_target, tau):
        for target_param, local_param in zip(dqn_target.parameters(), dqn_local.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)