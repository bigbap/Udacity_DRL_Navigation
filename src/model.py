import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self, state_space_n, action_space_n, seed=None):
        super(Model, self).__init__()
        if seed != None:
            self.seed = torch.manual_seed(seed)

        self.input = nn.Linear(state_space_n, 50)

        self.hidden1 = nn.Linear(50, 50)

        self.output = nn.Linear(50, action_space_n)
  
    def forward(self, x):
        x = F.leaky_relu(self.input(x))
        x = F.leaky_relu(self.hidden1(x))

        return self.output(x)

