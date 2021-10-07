#! /usr/bin/env python

import torch
from torch.distributions.categorical import Categorical
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
import gym
from PIL import Image

# DQN深度模型，用来估计Atari环境的Q函数
class DQN(nn.Module):

    def __init__(self, img_size, num_actions):
        super().__init__()

        # 输入图像的形状(c, h, w)
        self.img_size = img_size
        self.num_actions = num_actions

        # 对于Atari环境，输入为(4, 84, 84)
        self.featnet = nn.Sequential(
            nn.Conv2d(img_size[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        # 价值网络，根据特征输出每个动作的价值
        self.vnet = nn.Sequential(
            nn.Linear(self._feat_size(), 512),
            nn.ReLU(),
            nn.Linear(512, self.num_actions)
        )

    def _feat_size(self):
        with torch.no_grad():
            x = torch.randn(1, *self.img_size)
            x = self.featnet(x).view(1, -1)
        return x.size(1)

    def forward(self, x):        
        bs = x.size(0)

        # 提取特征
        feat = self.featnet(x).view(bs, -1)
        
        # 获取所有可能动作的价值
        values = self.vnet(feat)
        return values

    def act(self, x, epsilon=0.0):
        # ε-贪心算法
        if random.random() > epsilon:
            with torch.no_grad():
                values = self.forward(x)
            return values.argmax(-1).squeeze().item()
        else:
            return random.randint(0, self.num_actions-1)

# 策略网络，用于根据状态生成策略
class PolicyNet(nn.Module):

    def __init__(self, img_size, num_actions):
        super().__init__()

        # 输入图像的形状(c, h, w)
        self.img_size = img_size
        self.num_actions = num_actions

        # 对于Atari环境，输入为(4, 84, 84)
        self.featnet = nn.Sequential(
            nn.Conv2d(img_size[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )

        # 策略网络，计算每个动作的概率
        self.pnet = nn.Sequential(
            nn.Linear(self._feat_size(), 512),
            nn.ReLU(),
            nn.Linear(512, self.num_actions)
        )

    def _feat_size(self):
        with torch.no_grad():
            x = torch.randn(1, *self.img_size)
            x = self.featnet(x).view(1, -1)
        return x.size(1)

    def forward(self, x):
        feat = self.featnet(x).view(x.size(0), -1)
        return self.pnet(feat)

    def act(self, x):
        with torch.no_grad():
            logits = self(x)
            actions = Categorical(logits=logits).sample().squeeze()
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

class EnvWrapper(object):

    def __init__(self, env, num_frames):
        super().__init__()
        self.env_ = env
        self.num_frames = num_frames
        self.frame = deque(maxlen=num_frames)

    def _preprocess(self, img):
        # 预处理数据
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
    # 对经验回放的数据进行采样
    state, action, reward, next_state, done = buffer.sample(BATCH_SIZE)
    state = torch.tensor(state, dtype=torch.float32).cuda()
    reward = torch.tensor(reward, dtype=torch.float32).cuda()
    action = torch.tensor(action, dtype=torch.long).cuda()
    next_state = torch.tensor(next_state, dtype=torch.float32).cuda()
    done = torch.tensor(done, dtype=torch.float32).cuda()

    w1 = dqn1.state_dict()
    w2 = dqn2.state_dict()

    # 下一步状态的预测
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

    # 当前状态的预测
    predict1 = dqn1(state).gather(1, action.unsqueeze(-1)).squeeze()
    predict2 = dqn2(state).gather(1, action.unsqueeze(-1)).squeeze()
    lossv1 = 0.5*(predict1 - target).pow(2).mean()
    lossv2 = 0.5*(predict2 - target).pow(2).mean()

    # 损失函数的优化
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
    predict = (torch.min(predict1, predict2)*logits.softmax(-1)).sum(-1)
    lossp = (REG*dist.entropy() - predict).mean()
    
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
EPSILON_MIN = 0.01
EPSILON_MAX = 1.00
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
    next_state, reward, done, _ = env.step(action)
    buffer.push(state, action, reward, next_state, done)
    state = next_state
    episode_reward += reward

    if done:
        state = env.reset()
        all_rewards.append(episode_reward)
        episode_reward = 0

    if len(buffer) >= 10000:
        loss = train(buffer, pnet, dqn1, dqn2, optimizer)
