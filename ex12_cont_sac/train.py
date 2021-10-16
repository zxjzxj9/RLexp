#! /usr/bin/env python

import torch
from torch.distributions.categorical import Categorical
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
import gym
from PIL import Image

class DQN(nn.Module):

    def __init__(self, state_dim, act_dim):
        super().__init__()

        self.feat_net = nn.Sequential(
            nn.Linear(state_dim + act_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU()
        )

        self.vnet = nn.Linear(256, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], -1)
        feat = self.feat_net(x)
        value = self.vnet(feat)
        return value

    def val(self, state, action):
        with torch.no_grad():
            value = self(state, action)
            return value.squeeze().cpu().item()


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

class EnvWrapper(object):

    def __init__(self, env, num_frames):
        super().__init__()
        self.env_ = env
        self.num_frames = num_frames
        self.frame = deque(maxlen=num_frames)

    def _preprocess(self, img):
        img = Image.fromarray(img)
        img = img.convert("L")
        img = img.resize((84, 84))
        return np.array(img)/256.0

    def reset(self):
        obs = self.env_.reset()
        for _ in range(self.num_frames):
            self.frame.append(self._preprocess(obs))
        return np.stack(self.frame, 0)

    def step(self, action):
        obs, reward, done, _ = self.env_.step(action)
        self.frame.append(self._preprocess(obs))
        return np.stack(self.frame, 0), np.sign(reward), done, {}
    
    @property
    def env(self):
        return self.env_

def train(buffer, pnet, dqn1, dqn2, optimizer):
    state, action, reward, next_state, done = buffer.sample(BATCH_SIZE)
    state = torch.tensor(state, dtype=torch.float32).cuda()
    reward = torch.tensor(reward, dtype=torch.float32).cuda()
    action = torch.tensor(action, dtype=torch.float32).cuda()
    next_state = torch.tensor(next_state, dtype=torch.float32).cuda()
    done = torch.tensor(done, dtype=torch.float32).cuda()

    w1 = dqn1.state_dict()
    w2 = dqn2.state_dict()

    with torch.no_grad():
        logits = pnet(next_state)
        dist = Categorical(logits=logits)
        # acts = dist.sample()
        # target1 = dqn1(next_state).gather(1, acts.unsqueeze(-1))
        # target2 = dqn2(next_state).gather(1, acts.unsqueeze(-1))
        target1 = dqn1(next_state)
        target2 = dqn2(next_state)
        # print(torch.min(target1, target2).shape)
        # print(logits.softmax(-1).shape)
        # print(reward.shape)
        # print(dist.entropy().shape)
        target = reward + (1-done)*GAMMA*(
            (torch.min(target1, target2)*logits.softmax(-1)).sum(-1)
            - REG*dist.entropy())

    predict1 = dqn1(state).gather(1, action.unsqueeze(-1)).squeeze()
    predict2 = dqn2(state).gather(1, action.unsqueeze(-1)).squeeze()
    lossv1 = 0.5*(predict1 - target).pow(2).mean()
    lossv2 = 0.5*(predict2 - target).pow(2).mean()

    optimizer.zero_grad()
    lossv1.backward()
    lossv2.backward()
    torch.nn.utils.clip_grad_norm_(dqn1.parameters(), 0.5)
    torch.nn.utils.clip_grad_norm_(dqn2.parameters(), 0.5)
    optimizer.step()

    logits = pnet(state)
    dist = Categorical(logits=logits)
    # acts = dist.sample()
    with torch.no_grad():
        predict1 = dqn1(state)
        predict2 = dqn2(state)
        predict = REG*logits.log_softmax() - torch.min(predict1, predict2)
    lossp = (predict*logits.softmax(-1)).sum(-1)
    
    optimizer.zero_grad()
    lossp.backward()
    torch.nn.utils.clip_grad_norm_(pnet.parameters(), 0.5)
    optimizer.step()

    w1new = dqn1.state_dict()
    w2new = dqn2.state_dict()
    for k in w1.keys():
        w1[k].copy_(RHO*w1[k] + (1-RHO)*w1new[k])
        w2[k].copy_(RHO*w2[k] + (1-RHO)*w2new[k])
    dqn1.load_state_dict(w1)
    dqn2.load_state_dict(w2)
    return lossp.cpu().item()

GAMMA = 0.99
NFRAMES = 4
BATCH_SIZE = 32
NSTEPS = 4000000
NBUFFER = 100000
RHO = 0.995
REG = 0.02
env = gym.make('PongDeterministic-v4')
env = EnvWrapper(env, NFRAMES)

# print(env.reset().shape)
# print(env.step(1)[0].shape)

state = env.reset()
buffer = ExpReplayBuffer(NBUFFER)
pnet = PolicyNet((4, 84, 84), env.env.action_space.n)
dqn1 = DQN((4, 84, 84), env.env.action_space.n)
dqn2 = DQN((4, 84, 84), env.env.action_space.n)
pnet.cuda()
dqn1.cuda()
dqn2.cuda()
optimizer = torch.optim.Adam([
    {'params': pnet.parameters(), 'lr': 1e-4},
    {'params': dqn1.parameters(), 'lr': 1e-4},
    {'params': dqn2.parameters(), 'lr': 1e-4},
])

all_rewards = []
all_losses = []
episode_reward = 0

for nstep in range(NSTEPS):
    state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0).cuda()
    action = pnet.act(state_t)
    next_state, reward, done, _ = env.step((action, ))
    buffer.push(state, action, reward, next_state, done)
    state = next_state
    episode_reward += reward

    if done:
        state = env.reset()
        all_rewards.append(episode_reward)
        episode_reward = 0

    if len(buffer) >= 10000:
        loss = train(buffer, pnet, dqn1, dqn2, optimizer)
