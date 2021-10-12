#! /usr/bin/env python

import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import matplotlib.pyplot as plt
import random
from PIL import Image
import numpy as np
import tqdm
import copy
import gc
from collections import deque

class PolicyNet(nn.Module):

    def __init__(self, state_dim, act_dim, act_min, act_max):
        super().__init__()

        self.state_dim = state_dim
        self.act_dim = act_dim

        self.feat_net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU()
        )

        self.pnet_mu = nn.Linear(256, act_dim)
        # self.pnet_logs = nn.Linear(256, act_dim)
        self.pnet_logs = nn.Parameter(torch.zeros(act_dim))

        self.act_min = act_min
        self.act_max = act_max
    
    def forward(self, x):
        feat = self.feat_net(x)
        mu = 2.0*self.pnet_mu(feat).tanh()
        # sigma = (F.softplus(self.pnet_logs(feat)) + 1e-4).sqrt()
        sigma = (F.softplus(self.pnet_logs) + 1e-4).sqrt()
        return mu, sigma

    def act(self, x):
        with torch.no_grad():
            mu, sigma = self(x)
            dist = Normal(mu, sigma)
            return dist.sample()\
                .clamp(self.act_min, self.act_max)\
                .squeeze().cpu().item()

class ValueNet(nn.Module):

    def __init__(self, state_dim, act_dim):
        super().__init__()

        self.feat_net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU()
        )

        self.vnet = nn.Linear(256, 1)
    
    def forward(self, x):
        feat = self.feat_net(x)
        value = self.vnet(feat)
        return value

    def val(self, x):
        with torch.no_grad():
            value = self(x)
            return value.squeeze().cpu().item()


class ActionBuffer(object):

    def __init__(self, buffer_size):
        super().__init__()
        self.buffer = deque(maxlen=buffer_size)

    def reset(self):
        self.buffer.clear()

    def push(self, state, action, value, reward, done):
        self.buffer.append((state, action, value, reward, done))

    def sample(self, next_value):
        state, action, value, reward, done = \
            zip(*self.buffer)

        value = np.array(value + (next_value, ))
        done = np.array(done).astype(np.float32)
        reward = np.array(reward).astype(np.float32)
        delta = reward + GAMMA*(1-done)*value[1:] - value[:-1]

        rtn = np.zeros_like(delta).astype(np.float32)
        adv = np.zeros_like(delta).astype(np.float32)

        reward_t = next_value
        delta_t = 0.0

        # for i in reversed(range(len(reward))):
        #     reward_t = reward[i] + GAMMA*(1.0 - done[i])*reward_t
        #     delta_t = delta[i] + (GAMMA*LAMBDA)*(1.0 - done[i])*delta_t
        #     rtn[i] = reward_t
        #     adv[i] = delta_t
        
        rtn = reward + GAMMA*(1-done)*value[1:]
        # adv = (delta - np.mean(delta))/np.std(delta)
        adv = delta
        return np.stack(state, 0), np.stack(action, 0), rtn, adv

    def __len__(self):
        return len(self.buffer)


def train(buffer, next_value, pnet, vnet, optimizer):
    state, action, rtn, adv = buffer.sample(next_value)
    state = torch.tensor(state, dtype=torch.float32).cuda()
    action = torch.tensor(action, dtype=torch.float32).cuda()
    # print(action)
    # print(state.shape, action.shape)
    rtn = torch.tensor(rtn, dtype=torch.float32).cuda()
    adv = torch.tensor(adv, dtype=torch.float32).cuda()

    logits = pnet(state)
    values = vnet(state).squeeze()

    # 计算损失函数
    mu, sigma = pnet(state)
    dist = Normal(mu, sigma)
    
    # adv = (rtn - values).detach()
    # adv = (adv - adv.mean(-1))/adv.std()
    lossp = -(adv*dist.log_prob(action)).mean() - REG*dist.entropy().mean()
    lossv =  0.5*F.mse_loss(rtn, values)
    
    optimizer.zero_grad()
    lossp.backward()
    lossv.backward()
    torch.nn.utils.clip_grad_norm_(pnet.parameters(), 50.0)
    torch.nn.utils.clip_grad_norm_(vnet.parameters(), 50.0)
    optimizer.step()
    return lossp.item()


BATCH_SIZE = 16
NSTEPS = 1000000
GAMMA = 0.90
# LAMBDA = 0.95
# LAMBDA = 0.95
REG = 0.01
env = gym.make("Pendulum-v0")
buffer = ActionBuffer(BATCH_SIZE)
pnet = PolicyNet(env.observation_space.shape[0], env.action_space.shape[0], 
                 env.action_space.low[0], env.action_space.high[0])
vnet = ValueNet(env.observation_space.shape[0], env.action_space.shape[0])
pnet.cuda()
vnet.cuda()
optimizer = torch.optim.Adam([
    {'params': pnet.parameters(), 'lr': 1e-4},
    {'params': vnet.parameters(), 'lr': 1e-3},
])

all_rewards = []
all_losses = []
episode_reward = 0
loss = 0.0
# print(env.action_space.low, env.action_space.high)

state = env.reset()
for nstep in tqdm.tqdm(range(NSTEPS)):

    state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0).cuda()
    action = pnet.act(state_t)
    value = vnet.val(state_t)
    next_state, reward, done, _ = env.step((action,))
    buffer.push(state, action, value, 0.1*reward, done)
    state = next_state
    episode_reward += reward

    if done:
        state = env.reset()
        all_rewards.append(episode_reward)
        # print(episode_reward)
        # if episode_reward > -200: break
        episode_reward = 0

    if done or len(buffer) == BATCH_SIZE:
        # print("test\ntest\n")
        with torch.no_grad():
            state_t = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0).cuda()
            next_value = vnet.val(state_t)

        loss = train(buffer, next_value, pnet, vnet, optimizer)
        buffer.reset()
