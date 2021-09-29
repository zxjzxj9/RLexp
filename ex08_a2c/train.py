import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import matplotlib.pyplot as plt
import random
from PIL import Image
import numpy as np
import tqdm
import copy
import gc
from IPython.display import clear_output
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from collections import deque


torch.cuda.is_available()
print(torch.cuda.get_device_name(0))

NENV = 4
NUM_FRAMES = 4
NSTEPS = 1000000

class EnvWrapper(gym.Wrapper):

    def __init__(self, env, num_frames):
        super().__init__(env)
        gym.Wrapper.__init__(self, env)
        self.num_frames = num_frames
        self.frame = deque(maxlen=num_frames)
        self.observation_space = \
            gym.spaces.Box(0, 1, shape=(NUM_FRAMES, 84, 84), dtype=np.float32)

    def _preprocess(self, img):
        # 预处理数据
        img = Image.fromarray(img)
        img = img.convert("L")
        img = img.resize((84, 84))
        return np.array(img)/256.0

    def reset(self):
        obs = self.env.reset()
        for _ in range(self.num_frames):
            self.frame.append(self._preprocess(obs))
        return np.stack(self.frame, 0)

    def step(self, action):
        obs, reward, done, _ = self.env.step(action)
        self.frame.append(self._preprocess(obs))
        return np.stack(self.frame, 0), np.sign(reward), done, {}

    # @property
    # def state(self):
    #     return np.stack(self.frame, 0)

def create_env(rank):
    env = gym.make('PongDeterministic-v4')
    env.seed(rank)
    return EnvWrapper(env, NUM_FRAMES)


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

        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.orthogonal_(m.weight, gain=1.0)
                nn.init.zeros_(m.bias)

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

class ValueNet(nn.Module):
    def __init__(self, img_size):
        super().__init__()

        # 输入图像的形状(c, h, w)
        self.img_size = img_size

        # 对于Atari环境，输入为(4, 84, 84)
        self.featnet = nn.Sequential(
            nn.Conv2d(img_size[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )

        # 价值网络，根据特征输出每个动作的价值
        self.vnet = nn.Sequential(
            nn.Linear(self._feat_size(), 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.orthogonal_(m.weight, gain=1.0)
                nn.init.zeros_(m.bias)

    def _feat_size(self):
        with torch.no_grad():
            x = torch.randn(1, *self.img_size)
            x = self.featnet(x).view(1, -1)
        return x.size(1)

    def forward(self, x):
        feat = self.featnet(x).view(x.size(0), -1)
        return self.vnet(feat).squeeze(-1)

    def val(self, x):
        with torch.no_grad():
            values = self(x).squeeze()
        return values

class ActionBuffer(object):

    def __init__(self, buffer_size):
        super().__init__()
        self.buffer = deque(maxlen=buffer_size)

    def reset(self):
        self.buffer.clear()

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self):
        state, action, reward, next_state, done = \
            zip(*self.buffer)
        state = list(state)
        state.append(next_state[-1])

        return np.stack(state, 0), np.stack(action, 0), \
            np.stack(reward, 0), \
            1.0 - np.stack(done, 0).astype(np.float32)

    def __len__(self):
        return len(self.buffer)


def train(buffer, pnet, vnet, optimizer):
    states, actions, rewards, dones = buffer.sample()
    states = torch.tensor(states, dtype=torch.float32).cuda()
    actions = torch.tensor(actions.flatten(), dtype=torch.long).cuda()
    # rewards = torch.tensor(rewards, dtype=torch.float32).cuda()
    # dones = torch.tensor(dones, dtype=torch.float32).cuda()
    # print(states.shape, actions.shape, rewards.shape, dones.shape)
    # return 1.0

    nt, nvec, c, h, w = states.shape
    bs = nt - 1

    values = vnet(states.view(-1, c, h, w)).view(nt, nvec)

    val = values.detach().cpu().numpy()
    deltas = rewards + GAMMA*val[1:, :] - val[:-1, :]
    rtn = np.zeros_like(rewards)
    adv = np.zeros_like(rewards)
    rtn[-1, :] = rewards[-1, :] + GAMMA*dones[-1, :]*val[-1, :]
    adv[-1, :] = GAMMA*LAMBDA*dones[-1, :]*deltas[-1, :]
    for i in reversed(range(bs-1)):
        rtn[i, :] = rewards[i, :] + GAMMA*dones[i, :]*rtn[i+1, :]
        adv[i, :] = deltas[i, :] + GAMMA*LAMBDA*dones[i, :]*adv[i+1, :]

    rtn = torch.tensor(rtn.flatten(), dtype=torch.float32).cuda()
    adv = torch.tensor(adv.flatten(), dtype=torch.float32).cuda()

    values = values[:-1, :].flatten()
    logits = pnet(states[:-1, ...].reshape(-1, c, h, w))
    dist = Categorical(logits=logits)
    adv = (rtn - values).detach()
    # adv = (adv - adv.mean())/(adv.std() + 1e-6)
    lossp = -(adv*dist.log_prob(actions)).mean() - REG*dist.entropy().mean()
    lossv =  0.5*(rtn - values).pow(2).mean()
    
    optimizer.zero_grad()
    # lossp.backward()
    # lossv.backward()
    loss = lossp + lossv
    loss.backward()
    torch.nn.utils.clip_grad_norm_(pnet.parameters(), 0.5)
    torch.nn.utils.clip_grad_norm_(vnet.parameters(), 0.5)
    optimizer.step()
    # print(lossp, lossv)
    return lossp.cpu().item()

def evalenv(env, pnet, vnet):
    state = env.reset()
    val = None
    reward_tot = 0
    while True:
        state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0).cuda()
        action = pnet.act(state_t).cpu().item()
        if val is None: val = vnet.val(state_t).cpu().item()
        next_state, reward, done, _ = env.step(action)
        reward_tot += reward
        state = next_state
        if done: break
    return reward_tot, val

if __name__ == "__main__":
    GAMMA = 0.99
    LAMBDA = 0.95
    NFRAMES = 4
    BATCH_SIZE = 6
    NENV = 16
    NSTEPS = 10000000
    REG = 0.01
    # env = gym.make('Pong-v0')
    # env = gym.make('PongNoFrameskip-v4')
    env = EnvWrapper(gym.make('PongDeterministic-v4'), NFRAMES)
    venv = SubprocVecEnv([lambda: create_env(rank) for rank in range(NENV)])
    buffer = ActionBuffer(BATCH_SIZE)
    
    pnet = PolicyNet((4, 84, 84), env.env.action_space.n)
    vnet = ValueNet((4, 84, 84))
    pnet.cuda()
    vnet.cuda()
    optimizer = torch.optim.RMSprop([
        {'params': pnet.parameters(), 'lr': 7e-4, 'alpha': 0.99, 'eps': 1e-5},
        {'params': vnet.parameters(), 'lr': 7e-4, 'alpha': 0.99, 'eps': 1e-5},
    ])
    
    all_rewards = []
    all_losses = []
    all_values = []
    episode_reward = 0
    loss = 0.0
    
    env = EnvWrapper(gym.make('PongDeterministic-v4'), NFRAMES)
    
    state = venv.reset()
    
    #for nstep in tqdm.tqdm(range(NSTEPS)):
    for nstep in range(NSTEPS):
    
        state_t = torch.tensor(state, dtype=torch.float32).cuda()
        action = pnet.act(state_t).cpu()
        next_state, reward, done, _ = venv.step(action)
        buffer.push(state, action, reward, next_state, done)
        state = next_state
    
        if len(buffer) == BATCH_SIZE:
            loss = 0.9*loss + 0.1*train(buffer, pnet, vnet, optimizer)
            buffer.reset()
            # break
            # loss = train(venv, pnet, vnet, optimizer)
        
        if (nstep + 1) % 1000 == 0:
            reward, v0 = evalenv(env, pnet, vnet)
            all_rewards.append(reward)
            all_losses.append(loss)
            all_values.append(v0)
            print("nstep {:8d}, Reward {:12.6f}, loss {:12.6f}, value {:12.6f}".format(nstep, reward, loss, v0))
            # clear_output(True)
            # plt.figure(figsize=(20,5))
            # plt.subplot(131)
            # plt.plot(all_rewards)
            # plt.subplot(132)
            # plt.plot(all_losses)
            # plt.subplot(133)
            # plt.plot(all_values)
            # plt.show()
