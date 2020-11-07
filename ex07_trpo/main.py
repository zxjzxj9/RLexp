#! /usr/bin/env python

import queue
import matplotlib.pyplot as plt
from PIL import Image
import torch
from torchvision import transforms
import torch.nn.functional as F
import gym
import numpy as np
from torch import optim

from model import PolicyNet, EnvSampler
import torch.multiprocessing as mp

from torch.utils.tensorboard import SummaryWriter

mp.set_start_method('spawn', force=True)

BATCHSIZE = 4
def sample(sampler, queue, event):
    # print("HERE")
    while True:
        queue.put(sampler.sample_n(BATCHSIZE))
        event.wait()

def train_step(queue, event):

    # workers = []
    observ_batch = []
    action_batch = []
    discount_reward_batch = []
    avg_reward_batch = []

    for i in range(NWORKERS):
        observ, action, discount_reward, avg_reward = queue.get()
        observ_batch.append(observ)
        action_batch.append(action)
        discount_reward_batch.append(discount_reward)
        avg_reward_batch.append(avg_reward)
        del observ, action, discount_reward, avg_reward
        # worker.join()
    with torch.no_grad():
        observ_batch = torch.cat(observ_batch, 0)
        action_batch = torch.cat(action_batch, 0)
        discount_reward_batch = torch.cat(discount_reward_batch, 0)
        avg_reward_batch = torch.stack(avg_reward_batch, 0)
        avg_reward = avg_reward_batch.mean()

    # print(observ_batch.shape, action_batch.shape, discount_reward_batch.shape, avg_reward_batch.shape)
    # for _ in range(NWORKERS):
    #    print(queue.get())

    #import sys; sys.exit()
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
    event.set()

    global global_step
    global_step += 1
    # writer.add_scalar("Episode", sampler.episodes, global_step=global_step)
    writer.add_scalar("Reward", avg_reward.item(), global_step=global_step)
    writer.add_scalar("Total Loss", loss.item(), global_step=global_step)
    writer.add_scalar("Average Value", value_pr.mean().item(), global_step=global_step)
    del observ_batch, action_batch, discount_reward_batch, avg_reward_batch
    return avg_reward.item(), loss.item()

if __name__ == "__main__":
    writer = SummaryWriter("./log")

    env = gym.make("Pong-v0")
    MAXSTEP = 6
    NWORKERS = 4
    EPOCHSTEP = 4000*1024//(MAXSTEP*BATCHSIZE*NWORKERS) # around ~4000 1 EPOCH in A3C paper
    print("1 epoch contains {} steps".format(EPOCHSTEP))
    NEPOCH = 100*EPOCHSTEP
    GAMMA = 0.99
    NFRAMES = 4

    policy_net = PolicyNet(NFRAMES)
    policy_net.cuda()
    policy_net.share_memory() # make it store in shared memory
    opt = optim.RMSprop(policy_net.parameters(), lr=5e-4, alpha=0.99, eps=1e-5)

    samplers = [EnvSampler(env, policy_net, NFRAMES, MAXSTEP, GAMMA) for _ in range(NWORKERS)]
    global_step = 0

    ctx = mp.get_context('spawn')
    queue = ctx.Queue()
    event = ctx.Event()

    workers = []
    for i in range(NWORKERS):
        worker = ctx.Process(target=sample, args=(samplers[i], queue, event), daemon=True)
        worker.start()
        workers.append(worker)

    for i in range(NEPOCH):
        print("Step: {:6d}, Avg_reward: {:12.6f}, PolicyNet Loss: {:12.6f}".format(i+1, *train_step(queue, event)), end="\r")
        if (i+1) % EPOCHSTEP == 0:  torch.save({"policy": policy_net.state_dict()}, "model.pt")