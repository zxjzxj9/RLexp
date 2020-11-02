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

MAXSTEP = 128
BATCHSIZE = 16
EPOCH = 1000000
GAMMA = 0.99

NFRAMES = 4

policy_net = PolicyNet()
value_net = ValueNet()

policy_net.cuda()
value_net.cuda()
opt1 = optim.Adam(policy_net.parameters(), lr=1e-3)
opt2 = optim.Adam(value_net.parameters(), lr=1e-3)

global_step = 0
def train_step():

    observ_batch = []
    reward_batch = []
    action_batch = []
    discount_reward_batch = []

    # policy_net.cpu()
    # value_net.cpu()

    # mask =[]
    for _ in range(BATCHSIZE):
        observ = []
        reward = []
        action = []

        obs = env.reset()
        windows = [obs.copy()]*4
        observ.append(list_to_tensor(windows))

        for _ in range(MAXSTEP):
            with torch.no_grad():
                logits = policy_net(observ[-1].unsqueeze(0).cuda())
                dist = distributions.Categorical(logits=logits)
                act = dist.sample().cpu()
                action.append(act.item())
            obs, rwd, end, _ = env.step(act.item())
            windows.pop(0)
            windows.append(obs.copy())
            observ.append(list_to_tensor(windows))
            reward.append(rwd)
            if end: break

        # plt.imshow(np.array(obs), cmap='gray')
        # plt.savefig("test.png")
        # import sys; sys.exit()

        with torch.no_grad():
            observ = torch.stack(observ[:-1], dim=0).cuda()
            reward = torch.tensor(reward, dtype=torch.float32).cuda()
            action = torch.tensor(action).cuda()

            discount_reward = reward.clone()
            for idx in reversed(range(reward.size(0)-1)):
                discount_reward[idx] += GAMMA*discount_reward[idx+1]

        observ_batch.append(observ)
        action_batch.append(action)
        reward_batch.append(reward)
        discount_reward_batch.append(discount_reward)

    with torch.no_grad():
        observ_batch = torch.cat(observ_batch, dim=0)
        action_batch = torch.cat(action_batch, dim=0)
        reward_batch = torch.cat(reward_batch, dim=0)
        discount_reward_batch = torch.cat(discount_reward_batch, dim=0)
    # print(observ_batch.shape, reward_batch.shape, action_batch.shape)
    # import sys; sys.exit()

    avg_reward = reward_batch.mean()
    # print(avg_reward)

    opt1.zero_grad()
    opt2.zero_grad()

    policy_pr = policy_net(observ_batch)
    pr = policy_pr.softmax(dim=-1)
    value_pr = value_net(observ_batch)

    with torch.no_grad():
        advantage = (discount_reward_batch - value_pr).detach()
    # print(policy_pr.shape, action_batch.shape)
    # import sys; sys.exit()
    logprob = F.cross_entropy(policy_pr, action_batch, reduction='none')
    loss1 = (logprob * advantage + 0.01*(pr*pr.log()).sum(-1)).mean()
    # loss1 = (logprob * advantage).mean()
    loss2 = (value_pr - discount_reward_batch).pow(2).mean()
    loss1.backward()
    loss2.backward()

    opt1.step()
    opt2.step()

    global global_step
    global_step += 1
    writer.add_scalar("Reward", avg_reward.item(), global_step=global_step)
    writer.add_scalar("Policy Loss", loss1.item(), global_step=global_step)
    writer.add_scalar("Value Loss", loss2.item(), global_step=global_step)

    return avg_reward.item(), loss1.item(), loss2.item()

if __name__ == "__main__":
    for i in range(EPOCH):
        print("Epoch: {:6d}, Avg_reward: {:12.6f}, PolicyNet Loss: {:12.6f}, ValueNet Loss: {:12.6f}".format(i+1, *train_step()), end="\r")

    torch.save({"policy": policy_net.state_dict(), 
                "value": value_net.state_dict()}, "model.pt")