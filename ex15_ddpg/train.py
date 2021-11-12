#! /usr/bin/env python

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Independent
from collections import deque
import random
import numpy as np
import gym

class PolicyNet(nn.Module):

    def __init__(self, state_dim, act_dim, init_w=3e-3):
        super().__init__()

        self.state_dim = state_dim
        self.act_dim = act_dim

        self.featnet = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )

        self.pnet_mu = nn.Linear(256, act_dim)
        self.pnet_logs = nn.Linear(256, act_dim)
    
    def forward(self, x):
        feat = self.featnet(x)
        mu = self.pnet_mu(feat)
        sigma = self.pnet_logs(feat).clamp(-20, 2).exp()
        return Independent(Normal(loc=mu, scale=sigma), reinterpreted_batch_ndims=1)
    
    def sample_action(self, x, reparam=False):
        mu_given_s = self(x)
        u = mu_given_s.rsample() if reparam else mu_given_s.sample()
        a = torch.tanh(u)
        return a

    def act(self, x):
        with torch.no_grad():
            a, _ = self.sample_action_and_compute_log_pi(x)
            return a.cpu().item()

class DQN(nn.Module):

    def __init__(self, state_dim, act_dim, init_w=3e-3):
        super().__init__()

        self.featnet = nn.Sequential(
            nn.Linear(state_dim+act_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )

        self.vnet = nn.Linear(256, 1)
    
    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        feat = self.featnet(x)
        value = self.vnet(feat)
        return value

    def val(self, state, action):
        with torch.no_grad():
            value = self(state, action)
            return value.squeeze().cpu().item()


class VNet(nn.Module):
    def __init__(self, state_dim, init_w=3e-3):
        super().__init__()

        self.featnet = nn.Sequential(
            nn.Linear(state_dim , 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU()
        )

        self.vnet = nn.Linear(256, 1)
    
    def forward(self, x):
        feat = self.featnet(x)
        value = self.vnet(feat)
        return value

    def val(self, x):
        with torch.no_grad():
            value = self(x)
            return value.squeeze().cpu().item()

    def update(self, other, polyak=0.995):
        with torch.no_grad():
            for param1, param2 in zip(self.parameters(), other.parameters()):
                param1.data.copy_(polyak*param1.data+(1-polyak)*param2.data)
