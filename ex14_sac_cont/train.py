#! /usr/bin/env python

import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Independent
import matplotlib.pyplot as plt
import random
from PIL import Image
import numpy as np
import tqdm
import copy
import gc
from collections import deque

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
        return mu, sigma

    def act(self, x):
        with torch.no_grad():
            mu, sigma = self(x)
            dist = Normal(mu, sigma)
            return dist.sample().tanh().cpu().item()

class DQN(nn.Module):

    def __init__(self, state_dim, act_dim, init_w=3e-3):
        super().__init__()

        self.featnet = nn.Sequential(
            nn.Linear(state_dim + act_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )

        self.vnet = nn.Linear(256, 1)
        self.vnet.weight.data.uniform_(-init_w, init_w)
        self.vnet.bias.data.uniform_(-init_w, init_w)
    
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
        self.vnet.weight.data.uniform_(-init_w, init_w)
        self.vnet.bias.data.uniform_(-init_w, init_w)
    
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

class TwinDQN(nn.Module):
    def __init__(self, state_dim, act_dim):
        super().__init__()
        self.dqn1 = DQN(state_dim, act_dim)
        self.dqn2 = DQN(state_dim, act_dim)

    def update(self, other, polyak=0.995):
        with torch.no_grad():
            for param1, param2 in zip(self.parameters(), other.parameters()):
                param1.data.copy_(polyak*param1.data+(1-polyak)*param2.data)
