#! /usr/bin/env python

import gym
from model import PolicyNet, ValueNet

env = gym.make("CartPole-v1")
observation = env.reset()
print(observation)
print(env.observation_space)

MAXSTEP = 20
BATCHSIZE = 16

