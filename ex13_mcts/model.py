#! /usr/bin/env python

import torch
import torch.nn as nn


class PVNet(nn.Module):
    def __init__(self, in_chan, mid_chan, out_chan):
        super().__init__()

        mods = [nn.Conv2d(in_chan, mid_chan[0], 3, 1), nn.ReLU(inplace=True)]
        for cin, cout in zip(min_chan[:-1], [1:]):
            mods.append(nn.Conv2d(cin, cout, 3, 1)
            mods.append(nn.ReLU(inplace=True))
        mods.append(nn.Conv2d(mid_chan[-1], out_chan, 3, 1))
        mods.append(nn.ReLU(inplace=True))
        self.featnet = nn.Sequential(*mods) 
        self.policy = None
        self.value = None

    def forward(self, x, mask):
        pass

