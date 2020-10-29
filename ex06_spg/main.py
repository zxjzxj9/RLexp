#! /usr/bin/env python

import gym
from model import PolicyNet, ValueNet
import torch
from torch import optim, distributions


env = gym.make("CartPole-v1")
observation = env.reset()
print(observation)
print(env.observation_space)

MAXSTEP = 20
BATCHSIZE = 16
EPOCH = 200
GAMMA = 0.01

policy_net = PolicyNet()
value_net = ValueNet()

opt1 = optim.Adam(policy_net.parameters(), lr=1e-3)
opt2 = optim.Adam(value_net.parameters(), lr=1e-3)

# train one epoch
def train_step():

    observ_batch = []
    reward_batch = []
    action_batch = []

    for _ in range(BATCHSIZE):
        observ = []
        reward = []
        action = []
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
            if end: break
        print(observ, reward, action)
        import sys; sys.exit()
        observ_batch.append(observ)
        reward_batch.append(reward)
        action_batch.append(action)


if __name__ == "__main__":
    print(train_step())


