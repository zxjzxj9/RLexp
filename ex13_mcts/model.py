#! /usr/bin/env python

import torch
import torch.nn as nn


class PVNet(nn.Module):
    def __init__(self, h, w, channs, nactions):
        super().__init__()

        mods = []
        for cin, cout in zip(channs[:-1], channs[1:]):
            mods.append(nn.Conv2d(cin, cout, 3, 1)
            mods.append(nn.ReLU(inplace=True))
        mods.append(nn.Flatten())
        self.featnet = nn.Sequential(*mods) 
        self.policy = nn.Linear(h*w*nchanns[-1], nactions)
        self.value = nn.Linear(h*w*nchanns[-1], 1)

    def forward(self, x, mask):
        pass

