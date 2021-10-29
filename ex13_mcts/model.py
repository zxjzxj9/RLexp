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
        self.pnet = nn.Linear(h*w*nchanns[-1], nactions)
        self.vnet = nn.Linear(h*w*nchanns[-1], 1)

    def forward(self, x, mask=None):
        bs = x.size(0)
        feat = self.featnet(x).view(bs, -1)
        logits = self.pnet(feat)
        if mask is not None:
            logits[mask] = -float('inf')
        value = self.vnet(feat)
        return logits, value

if __name__ == "__main__":
    pass
