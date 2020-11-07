#! /usr/bin/env python

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from torch import optim, distributions

trans = transforms.Compose([
    transforms.Resize((110, 84)),
    transforms.CenterCrop(84),
    transforms.ToTensor()
])

def numpy_to_tensor(x):
    x = Image.fromarray(x)
    x = trans(x)
    return x.mean(0)

def list_to_tensor(li):
    li = [numpy_to_tensor(l) for l in li]
    with torch.no_grad():
        li = torch.stack(li, 0)
    return li

class PolicyNet(nn.Module):
    def __init__(self, nframe=4, nspace=6):

        super().__init__()
        self.net1 = nn.Sequential(
            nn.Conv2d(nframe, 16, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(16, 32, 4, stride=2),
            nn.ReLU(),
        )
        self.net2 = nn.Sequential(
             nn.Linear(32*9*9, 256),
             nn.ReLU(),
        )
        self.pnet = nn.Linear(256, nspace)
        self.vnet = nn.Linear(256, 1)

    def forward(self, x):
        x = self.net1(x)
        x = x.view(x.size(0), -1)
        x = self.net2(x)
        p = self.pnet(x)
        v = self.vnet(x).squeeze(-1)
        return p, v

class EnvSampler(object):

    def __init__(self, env, pnet, nframes=4, maxstep=128, gamma=0.99):

        self.env = env
        self.pnet = pnet
        self.nframes = nframes
        self.maxstep = maxstep
        self.gamma = gamma
        self.neps = 0 # number of episodes
        self.reset()

    @property
    def episodes(self):
        return self.neps

    def reset(self):
        obs = self.env.reset()
        self.windows = [obs.copy()]*self.nframes

    def sample(self):

        observ_batch = []
        reward_batch = []
        action_batch = []
        discount_reward_batch = []
        avg_reward = 0.0
        last_reward = 0.0
        end = False

        observ = [list_to_tensor(self.windows)]
        action = []
        reward = []

        for _ in range(self.maxstep):
            with torch.no_grad():
                # print(observ[-1].shape)
                logits, last_reward = self.pnet(observ[-1].unsqueeze(0).cuda())
                dist = distributions.Categorical(logits=logits)
                act = dist.sample().cpu()
                action.append(act.item())

            # obs = None
            # rwd = 0.0
            # for _ in range(self.nframes):
            #     obs, r, end, _ = self.env.step(act.item())
            #     rwd += r
            obs, rwd, end, _ = self.env.step(act.item())

            self.windows.pop(0)
            self.windows.append(obs.copy())
            observ.append(list_to_tensor(self.windows))
            reward.append(rwd)
            if end: break

        observ = torch.stack(observ[:-1], dim=0).cuda()
        reward = torch.tensor(reward, dtype=torch.float32).cuda()
        action = torch.tensor(action).cuda()
        avg_reward = reward.mean()

        if not end:
            with torch.no_grad():
                # print(reward.shape)
                # print(last_reward.shape)
                # print(reward[-1], reward[-2])
                # import sys; sys.exit()
                reward[-1] = reward[-1] + self.gamma*last_reward.squeeze(0)

        with torch.no_grad():
            discount_reward = reward.clone()
            for idx in reversed(range(reward.size(0)-1)):
                discount_reward[idx] += self.gamma*discount_reward[idx+1]

        if end:
            self.neps += 1
            self.reset()

        return observ, action, discount_reward, avg_reward

    def sample_n(self, n=1):
        observ_batch = []
        action_batch = []
        discount_reward_batch = []
        avg_reward_batch = []

        with torch.no_grad():
            for _ in range(n):
                observ, action, discount_reward, avg_reward = self.sample()
                observ_batch.append(observ)
                action_batch.append(action)
                discount_reward_batch.append(discount_reward)
                avg_reward_batch.append(avg_reward)

            observ_batch = torch.cat(observ_batch, 0)
            action_batch = torch.cat(action_batch, 0)
            discount_reward_batch = torch.cat(discount_reward_batch, 0)
            avg_reward_batch = torch.stack(avg_reward_batch, 0).mean()
        return observ_batch, action_batch, discount_reward_batch, avg_reward_batch


if __name__ == "__main__":
    pass