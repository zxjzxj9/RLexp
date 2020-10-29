#! /usr/bin/env python

import gym
from model import PolicyNet, ValueNet
import torch
from torch import optim, distributions
import torch.nn.functional as F

env = gym.make("CartPole-v1")
# observation = env.reset()
# print(observation)
# print(env.observation_space)

MAXSTEP = 100
BATCHSIZE = 16
EPOCH = 1000
GAMMA = 0.99

policy_net = PolicyNet()
value_net = ValueNet()

policy_net.cuda()
value_net.cuda()
opt1 = optim.Adam(policy_net.parameters(), lr=1e-3)
opt2 = optim.Adam(value_net.parameters(), lr=1e-3)

# train one epoch
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
        observ.append(obs.copy())
        for _ in range(MAXSTEP):
            with torch.no_grad():
                logits = policy_net(torch.tensor(obs, dtype=torch.float32))
                dist = distributions.Categorical(logits=logits)
                act = dist.sample()
                action.append(act.item())
            obs, rwd, end, _ = env.step(act.item())
            observ.append(obs.copy())
            reward.append(rwd)
            mask.append(1)
            if end: break
        #print(torch.tensor(observ).shape, reward, action)
        #import sys; sys.exit()
        observ_batch.append(torch.tensor(observ, dtype=torch.float32))
        reward_batch.append(torch.tensor(reward, dtype=torch.float32))
        action_batch.append(torch.tensor(action))
        mask_batch.append(torch.tensor(mask))
    observ_batch = torch.nn.utils.rnn.pad_sequence(observ_batch, batch_first=True).cuda()
    reward_batch = torch.nn.utils.rnn.pad_sequence(reward_batch, batch_first=True).cuda()
    action_batch = torch.nn.utils.rnn.pad_sequence(action_batch, batch_first=True).cuda()
    mask_batch = torch.nn.utils.rnn.pad_sequence(mask_batch, batch_first=True).cuda()
    avg_reward = reward_batch.mean()
    # print(observ_batch.shape, reward_batch.shape, action_batch.shape)
    # print(mask_batch)
    policy_net.cuda()
    value_net.cuda()

    with torch.no_grad():
        discount_reward_batch = reward_batch.clone()
        for idx in reversed(range(reward_batch.size(1)-1)):
            discount_reward_batch[:, idx] += GAMMA*discount_reward_batch[:, idx+1]

    opt1.zero_grad()
    opt2.zero_grad()
    observ_batch = observ_batch[:, :-1, :]

    policy_pr = policy_net(observ_batch).transpose(1, 2)
    value_pr = value_net(observ_batch)

    advantage = (discount_reward_batch - value_pr).detach()
    logprob = F.cross_entropy(policy_pr, action_batch, reduction='none')
    loss1 = (logprob * mask_batch * advantage).mean()
    loss2 = ((value_pr - discount_reward_batch)*mask_batch).pow(2).mean()
    loss1.backward()
    loss2.backward()

    opt1.step()
    opt2.step()
    return avg_reward.item(), loss1.item(), loss2.item()


if __name__ == "__main__":
    for i in range(EPOCH):
        print("Avg_reward: {:12.6f}, PolicyNet Loss: {:12.6f}, ValueNet Loss: {:12.6f}".format(*train_step()), end="\r")
    torch.save({"policy": policy_net.state_dict(), 
                "value": value_net.state_dict()}, "model.pt")