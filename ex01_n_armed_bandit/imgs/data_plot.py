#! /usr/bin/env python

import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np
sns.set(style="darkgrid")


greedy = np.genfromtxt("./greed.dat", dtype=(np.int32, np.float, np.float))
#greedy = np.genfromtxt("./greed.dat", dtype=None)
#greedy = np.genfromtxt("./greed.dat", dtype=[('f0', '<i4'), ('f1', '<f8'), ('f2', '<f8')])
idx = [t[0] for t in greedy]
reward = [t[1] for t in greedy]
prob = [t[2] for t in greedy]
#print(idx, reward, prob)
plt.plot(idx, reward, color='r', ls="--", label="greedy_reward")
plt.plot(idx, prob, color='r', ls="-", label="greedy_prob")

greedy = np.genfromtxt("./egreed.dat", dtype=(np.int32, np.float, np.float))
idx = [t[0] for t in greedy]
reward = [t[1] for t in greedy]
prob = [t[2] for t in greedy]
plt.plot(idx, reward, color='b', ls="--", label="$\epsilon$-greedy_reward")
plt.plot(idx, prob, color='b', ls="-", label="$\epsilon$-greedy_prob")


greedy = np.genfromtxt("./grad.dat", dtype=(np.int32, np.float, np.float))
idx = [t[0] for t in greedy]
reward = [t[1] for t in greedy]
prob = [t[2] for t in greedy]
plt.plot(idx, reward, color='g', ls="--", label="gradient_reward")
plt.plot(idx, prob, color='g', ls="-", label="gradient_prob")

plt.legend()
plt.xlabel("step")
plt.savefig("result.png", dpi=300)
plt.show()
