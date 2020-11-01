#! /usr/bin/env python

import matplotlib.pyplot as plt
from PIL import Image
import torch
from torchvision import transforms
import torch.nn.functional as F
import gym
import numpy as np
from torch import optim, distributions

from model import PolicyNet, ValueNet, list_to_tensor
import queue

env = gym.make("Pong-v0")

# obs = env.reset()
# obs = Image.fromarray(obs)
# obs = transforms.Resize((110, 84))(obs)
# obs = transforms.CenterCrop(84)(obs)
# obs = F.interpolate(torch.tensor(obs).permute(2, 0, 1), (3, 110, 84)).numpy()
# print(obs.shape)
# print(env.action_space)
# plt.imshow(np.array(obs), cmap='gray')
# plt.savefig("test.png")

MAXSTEP = 100
BATCHSIZE = 16
EPOCH = 1000
GAMMA = 0.99

NFRAMES = 4

policy_net = PolicyNet()
value_net = ValueNet()

policy_net.cuda()
value_net.cuda()
opt1 = optim.Adam(policy_net.parameters(), lr=1e-3)
opt2 = optim.Adam(value_net.parameters(), lr=1e-3)

def train_step():

    observ_batch = []
    reward_batch = []
    action_batch = []
    mask_batch = []

    policy_net.cpu()
    value_net.cpu()
    for _ in range(BATCHSIZE):
        observ = []
        reward = []
        action = []
        mask =[]
        obs = env.reset()
        windows = [obs.copy()]*4
        observ.append(list_to_tensor(windows))

        for _ in range(MAXSTEP):
            with torch.no_grad():
                logits = policy_net(observ[-1].unsqueeze(0))
                dist = distributions.Categorical(logits=logits)
                act = dist.sample()
                action.append(act.item())
            obs, rwd, end, _ = env.step(act.item())
            windows.pop(0)
            windows.append(obs.copy())
            observ.append(list_to_tensor(windows))
            reward.append(rwd)
            mask.append(1)
            if end: break

        observ_batch.append(torch.
        reward_batch.append(torch.tensor(reward))
        action_batch.append(torch.tensor(action))
        mask_batch.append(torch.tensor(mask))

        observ_batch = torch.nn.utils.rnn.pad_sequence(observ_batch, batch_first=True).cuda()
        reward_batch = torch.nn.utils.rnn.pad_sequence(reward_batch, batch_first=True).cuda()
        action_batch = torch.nn.utils.rnn.pad_sequence(action_batch, batch_first=True).cuda()
        mask_batch = torch.nn.utils.rnn.pad_sequence(mask_batch, batch_first=True).cuda()
        avg_reward = reward_batch.mean()

        print(avg_reward)


if __name__ == "__main__":
    train_step()