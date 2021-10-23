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
            nn.Flatten()
        )

        gain = nn.init.calculate_gain('relu')  
        self.vnet1 = nn.Sequential(
            nn.Linear(self._feat_size(), 512),
            nn.ReLU()
        )
        self._init(self.featnet, gain)
        self._init(self.vnet1, gain)

        # 价值网络，根据特征输出每个动作的价值
        gain = 1.0
        self.vnet2 = nn.Linear(512, num_actions)
        self._init(self.vnet2, gain)
        

    def _feat_size(self):
        with torch.no_grad():
            x = torch.randn(1, *self.img_size)
            x = self.featnet(x).view(1, -1)
        return x.size(1)

    def _init(self, mod, gain):
        for m in mod.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.xavier_normal_(m.weight, gain=gain)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        feat = self.featnet(x)
        feat = self.vnet1(feat)
        return self.vnet2(feat).squeeze(-1)#.clamp(-10, 10)


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
            nn.Flatten(),
        )

        self.log_alpha = nn.Parameter(torch.tensor([0.0]))
        self.target_entropy = -0.98*np.log(1.0/num_actions)

        gain = nn.init.calculate_gain('relu')
        self.pnet1 = nn.Sequential(
            nn.Linear(self._feat_size(), 512),
            nn.ReLU(),
        )
        self._init(self.featnet, gain)
        self._init(self.pnet1, gain)
        
        # 策略网络，计算每个动作的概率
        gain = 1.0
        self.pnet2 = nn.Linear(512, self.num_actions)
        self._init(self.pnet2, gain)

    def _feat_size(self):
        with torch.no_grad():
            x = torch.randn(1, *self.img_size)
            x = self.featnet(x).view(1, -1)
        return x.size(1)

    def _init(self, mod, gain):
        for m in mod.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.xavier_normal_(m.weight, gain=gain)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        feat = self.featnet(x)
        feat = self.pnet1(feat)
        return self.pnet2(feat)

    def act(self, x):
        with torch.no_grad():
            logits = self(x)
            m = Categorical(logits=logits).sample().squeeze()
        return m.cpu().item()

class TwinnedQNetwork(nn.Module):
    def __init__(self, img_size, num_actions):
        super().__init__()
        self.dqn1 = DQN(img_size, num_actions)
        self.dqn2 = DQN(img_size, num_actions)

    def update(self, other, rho=0.995):
        param1 = self.state_dict()
        param2 = other.state_dict()

        for k in param1.keys():
            param1[k].copy_(rho*param1[k] + (1-rho)*param2[k])

        self.load_state_dict(param1)

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


def train(buffer, pnet, localnet, targetnet, optimizer):
    # 对经验回放的数据进行采样
    state, action, reward, next_state, done = buffer.sample(BATCH_SIZE)
    state = torch.tensor(state, dtype=torch.float32).cuda()
    reward = torch.tensor(reward, dtype=torch.float32).cuda()
    action = torch.tensor(action, dtype=torch.long).unsqueeze(-1).cuda()
    next_state = torch.tensor(next_state, dtype=torch.float32).cuda()
    done = torch.tensor(done, dtype=torch.float32).cuda()

    with torch.no_grad():
        alpha = pnet.log_alpha.exp().clamp(0.0, 1.0)
    # alpha = REG

    # 下一步状态的预测
    with torch.no_grad():
        logits = pnet(next_state)
        probs = logits.softmax(-1)
        target1 = targetnet.dqn1(next_state)
        target2 = targetnet.dqn2(next_state)
        qmin = torch.min(target1, target2)
        qval = ((qmin - alpha*probs.log())*probs).sum(-1)
        target = reward + (1-done)*GAMMA*qval

    # print(probs)
    # print(target, target.shape)
    # 当前状态的预测
    predict1 = localnet.dqn1(state)
    predict2 = localnet.dqn2(state)
    lossv1 = 0.5*(predict1.gather(1, action).squeeze() - target).pow(2).mean()
    lossv2 = 0.5*(predict2.gather(1, action).squeeze() - target).pow(2).mean()
    # print(predict1)
    # print(action.squeeze())
    # print(predict1.gather(1, action).squeeze())

    logits = pnet(state)
    probs = logits.softmax(-1)
    predict1 = predict1.detach()
    predict2 = predict2.detach()
    qmin = torch.min(predict1, predict2)
    target = (qmin*probs).sum(-1).mean()
    entropy = -(probs*probs.log()).sum(-1) #.mean()
    lossp = -alpha*entropy - target
    lossa = -torch.mean(pnet.log_alpha*(pnet.target_entropy - entropy.detach()))
    
    optimizer.zero_grad()
    lossv1.backward()
    lossv2.backward()
    lossp.backward()
    lossa.backward()
    # torch.nn.utils.clip_grad_norm_(dqn1.parameters(), 0.5)
    # torch.nn.utils.clip_grad_norm_(dqn2.parameters(), 0.5)
    # torch.nn.utils.clip_grad_norm_(pnet.parameters(), 0.5)
    optimizer.step()

    # targetnet.update(localnet, RHO)
    return entropy.mean().cpu().item(), alpha.cpu().item() #lossp.cpu().item()

GAMMA = 0.99
NFRAMES = 4
BATCH_SIZE = 32
NSTEPS = 4000000
NBUFFER = 100000
# RHO = 0.995
# REG = 0.02
env = gym.make('QbertDeterministic-v4')
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
ep_len = 0
for nstep in range(NSTEPS):
    state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0).cuda()
    action = pnet.act(state_t)
    next_state, reward, done, _ = env.step(action)
    if ep_len == env.env._max_episode_steps:
        done = False
    buffer.push(state, action, reward, next_state, done)
    state = next_state
    episode_reward += reward

    if done or ep_len == env.env._max_episode_steps:
        state = env.reset()
        all_rewards.append(episode_reward)
        episode_reward = 0
        ep_len = 0

    if len(buffer) >= 20000 and nstep%NUPDATE == 0:
        loss, alpha = train(buffer, pnet, localnet, targetnet, optimizer)

    if nstep%TUPDATE == 0:
        targetnet.load_state_dict(localnet.state_dict())
