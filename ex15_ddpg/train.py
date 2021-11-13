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
            a = self.sample_action(x)
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
        
    def update(self, other, polyak=0.995):
        with torch.no_grad():
            for param1, param2 in zip(self.parameters(), other.parameters()):
                param1.data.copy_(polyak*param1.data+(1-polyak)*param2.data)


class NormalizedActions(gym.ActionWrapper):
    def action(self, action):
        low  = self.action_space.low
        high = self.action_space.high
        
        action = low + (action + 1.0) * 0.5 * (high - low)
        action = np.clip(action, low, high)
        
        return action

    def reverse_action(self, action):
        low  = self.action_space.low
        high = self.action_space.high
        
        action = 2 * (action - low) / (high - low) - 1
        action = np.clip(action, low, high)
        
        return actions

from collections import deque
class ExpReplayBuffer(object):

    def __init__(self, buffer_size):
        super().__init__()
        self.buffer = deque(maxlen=buffer_size)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, bs):
        state, action, reward, next_state, done = \
            zip(*random.sample(self.buffer, bs))
        return np.stack(state, 0), np.stack(action, 0), \
            np.stack(reward, 0), np.stack(next_state, 0), \
            np.stack(done, 0).astype(np.float32)

    def __len__(self):
        return len(self.buffer)

def train(buffer, pnet, vnet, vnet_target, optim_p, optim_v):
    state, action, reward, next_state, done = buffer.sample(BATCH_SIZE)
    state = torch.tensor(state, dtype=torch.float32).cuda()
    reward = torch.tensor(reward, dtype=torch.float32).cuda().unsqueeze(-1)
    action = torch.tensor(action, dtype=torch.float32).cuda().unsqueeze(-1)
    next_state = torch.tensor(next_state, dtype=torch.float32).cuda()
    done = torch.tensor(done, dtype=torch.float32).cuda().unsqueeze(-1)

    with torch.no_grad():
        next_action = pnet.sample_action(next_state)
        next_qval = vnet_target(next_state, next_action)
        target = reward + GAMMA * (1 - done) * next_qval

    value = vnet(state, action)
    lossv = 0.5*(value - target).pow(2).mean()
    optim_v.zero_grad()
    lossv.backward()
    torch.nn.utils.clip_grad_value_(vnet.parameters(), 1.0)
    optim_v.step()

    # 计算损失函数
    for param in vnet.parameters():
        param.requires_grad = False

    action = pnet.sample_action(state, True)
    qval = vnet(state, action)
    lossp = -torch.mean(qval)
    optim_p.zero_grad()
    lossp.backward()
    torch.nn.utils.clip_grad_value_(pnet.parameters(), 1.0)
    optim_p.step()
    for param in vnet.parameters():
        param.requires_grad = True

    vnet_target.update(vnet)
    return lossp
