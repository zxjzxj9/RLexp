#! /usr/bin/env python

import torch.nn as nn

class PolicyNet(nn.Module):

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, 8),
            nn.ReLU(),
            nn.Linear(8, 8),
            nn.ReLU(),
            nn.Linear(8, 2)    
        )
    
    def forward(self, x):
        """ Input: state
            Output: action
        """
        return self.net(x)


class ValueNet(nn.Module):

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, 8),
            nn.ReLU(),
            nn.Linear(8, 8),
            nn.ReLU(),
            nn.Linear(8, 1)    
        )

    def forward(self, x):
        """ Input: state
            Output: value
        """
        return self.net(x).squeeze(-1)