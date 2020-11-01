#! /usr/bin/env python

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

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
    def __init__(self, nframe = 4, nspace=6):

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
             nn.Linear(256, nspace),
        )

    def forward(self, x):
        x = self.net1(x)
        x = x.view(x.size(0), -1)
        x = self.net2(x)
        return x

class ValueNet(nn.Module):
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
             nn.Linear(256, 1),
        )

    def forward(self, x):
        x = self.net1(x)
        x = x.view(x.size(0), -1)
        x = self.net2(x).squeeze(-1)
        return x


if __name__ == "__main__":
    pnet = ValueNet()
    img = torch.randn(32, 4, 84, 84)
    print(pnet(img).shape)