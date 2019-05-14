import gym
import torch
import random
import torch.nn as nn
import numpy as np
import copy

# simulate the Q value 
class DQN(nn.Module):
    def __init__(self, naction, nstate, nhidden):
        super(DQN, self).__init__()
        self.naction = naction
        self.nstate = nstate
        self.linear1 = nn.Linear(naction + nstate, nhidden)
        self.linear2 = nn.Linear(nhidden, nhidden)
        self.linear3 = nn.Linear(nhidden, 1)
    
    def forward(self, state, action):
        # action is in range [0, nstate)
        # state has a size nbatch x nstate
        action_enc = torch.zeros(action.size(0), self.naction).to(action.device)
        action_enc.scatter_(1, action.unsqueeze(-1), 1)
        output = torch.cat((state, action_enc), dim=-1)
        output = torch.relu(self.linear1(output))
        output = torch.relu(self.linear2(output))
        output = self.linear3(output)
        return output.squeeze(-1)
    
class HuberLoss(nn.Module):
    def __init__(self):
        super(HuberLoss, self).__init__()
    
    def forward(self, delta):
        delta = torch.abs(delta)
        mask = (delta < 1.0).float()
        loss = mask*0.5*delta**2 + (1-mask)*(delta-0.5)
        return loss.mean()

# given an action, return the possible Q for this action
dqn = DQN(2, 4, 8)
dqn_t = DQN(2, 4, 8)
dqn_t.load_state_dict(copy.deepcopy(dqn.state_dict()))

feed = (torch.randn(4, 4), torch.tensor([0, 1, 1, 1]))
# test the dqn and dqn_t network
print(dqn(*feed).argmax())
print(dqn_t(*feed).argmax())

# For replay during training
class Memory(object):
    def __init__(self, capacity=1000):
        #self.bs = bs
        self.capacity = capacity
        self.size = 0

        self.data = []
        
    def __len__(self):
        return self.size
        
    def push(self, state, action, state_next, reward, is_ended):
        
        if len(self) > self.capacity:
            k = random.randint(self.capacity)
            self.data.pop(k)
            self.size -= 1
        
        self.data.append((state, action, state_next, reward, is_ended))
        
    def sample(self, bs):
        data = random.choices(self.data, k=bs)
        states, actions, states_next, rewards, is_ended = zip(*data)
        
        # convert numpy array into tensor
        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions)
        states_next = torch.tensor(states_next, dtype=torch.float32)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        is_ended = torch.tensor(is_ended, dtype=torch.float32)
        
        return states, actions, states_next, rewards, is_ended
        
env = gym.make('CartPole-v0')
#env.render()

# use epsilon-greedy policy
eps = 0.1
gamma = 0.999

optim = torch.optim.Adam(dqn.parameters(), lr=1e-3)
criterion = HuberLoss()         
                      
step_cnt = 0

mem = Memory()

for episode in range(300):
    state = env.reset()
    #print("")
    #print("Episode {}".format(episode))
    # reward_tot = 0
    while True:
        action_t = torch.tensor([0, 1])
        state_t = torch.tensor([state, state], dtype=torch.float32)
        
        torch.set_grad_enabled(False)
        q_t = dqn(state_t, action_t)
        max_t = q_t.argmax()
        torch.set_grad_enabled(True)
        
        # take a epsilon greedy algorithm
        if random.random() < eps:
            max_t = random.choice([0, 1])
        else:
            max_t = max_t.item()
        
        state_next, reward, done, info = env.step(max_t)
        
        mem.push(state, max_t, state_next, reward, done)
        state = state_next
        
        if done:
            #print("End episode...")
            break
    
        # replay to train the policy network
        for _ in range(10):
            state_t, action_t, state_next_t, reward_t, is_ended_t = mem.sample(32)

            q1 = dqn(state_t, action_t)
            
            torch.set_grad_enabled(False)
            q2_0 = dqn_t(state_next_t, torch.zeros(state_t.size(0), dtype=torch.long))
            q2_1 = dqn_t(state_next_t, torch.ones(state_t.size(0), dtype=torch.long))
            q2_max = reward_t + gamma*(1-is_ended_t)*(torch.stack((q2_0, q2_1), dim=1).max(1)[0])
            #q2_max.clamp_(min=-1, max=1)
            torch.set_grad_enabled(True)
            
            delta = q2_max - q1
            loss = criterion(delta)
    
            optim.zero_grad()
            loss.backward()
            for p in dqn.parameters(): p.grad.data.clamp_(-1, 1)
            print("Episode {}, Current step {:06d}, Loss: {:.3f}".format(episode, step_cnt, loss.item()), end = "\r")
            optim.step()          
            step_cnt += 1
                            
            if step_cnt % 1000 == 0:
                dqn_t.load_state_dict(copy.deepcopy(dqn.state_dict()))
env.close()

### Test the model
state = env.reset()
step = 0
while True:
    env.render()
    action_t = torch.tensor([0, 1])
    state_t = torch.tensor([state, state], dtype=torch.float32)
    
    torch.set_grad_enabled(False)
    q_t = dqn(state_t, action_t)
    max_t = q_t.argmax()
    torch.set_grad_enabled(True)
    
    max_t = max_t.item()
    state_next, reward, done, info = env.step(max_t)
    state = state_next
    step += 1
    print(reward, q_t, max_t, step)
    if done: break
env.close()
