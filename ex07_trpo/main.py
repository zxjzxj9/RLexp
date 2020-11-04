#! /usr/bin/env python

import matplotlib.pyplot as plt
from PIL import Image
import torch
from torchvision import transforms
import torch.nn.functional as F
import gym
import numpy as np
from torch import optim, distributions

from model import PolicyNet, EnvSampler

from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter("./log")

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

MAXSTEP = 32
BATCHSIZE = 32
EPOCH = 4000*100 # around ~4000 1 EPOCH in A3C paper
GAMMA = 0.99

NFRAMES = 4

policy_net = PolicyNet(NFRAMES)

policy_net.cuda()
opt = optim.RMSprop(policy_net.parameters(), lr=1e-3, alpha=0.99, eps=1e-5)

sampler = EnvSampler(env, policy_net, NFRAMES, MAXSTEP, GAMMA)
global_step = 0

def train_step():

    observ_batch, action_batch, discount_reward_batch, avg_reward = sampler.sample_n(BATCHSIZE)

    opt.zero_grad()

    policy_pr, value_pr = policy_net(observ_batch)
    pr = policy_pr.softmax(-1)
    # print(policy_pr.shape, value_pr.shape, discount_reward_batch.shape)

    with torch.no_grad():
        advantage = (discount_reward_batch - value_pr).detach()
    logprob = F.cross_entropy(policy_pr, action_batch, reduction='none')
    # print(logprob.shape, advantage.shape)
    # import sys; sys.exit()
    loss = (logprob * advantage).mean() + 0.01*(pr*pr.log()).sum(-1).mean() + \
           0.5*(value_pr - discount_reward_batch).pow(2).mean()

    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 0.5)

    opt.step()

    global global_step
    global_step += 1
    writer.add_scalar("Episode", sampler.episodes, global_step=global_step)
    writer.add_scalar("Reward", avg_reward.item(), global_step=global_step)
    writer.add_scalar("Total Loss", loss.item(), global_step=global_step)
    writer.add_scalar("Average Value", value_pr.mean().item(), global_step=global_step)

    return avg_reward.item(), loss.item()

if __name__ == "__main__":
    for i in range(EPOCH):
        print("Epoch: {:6d}, Avg_reward: {:12.6f}, PolicyNet Loss: {:12.6f}".format(i+1, *train_step()), end="\r")
        if (i+1) % 4000 == 0:  torch.save({"policy": policy_net.state_dict()}, "model.pt")