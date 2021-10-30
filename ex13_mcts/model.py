#! /usr/bin/env python

import torch
import torch.nn as nn


class PVNet(nn.Module):
    def __init__(self, h, w, nchanns, nactions):
        super().__init__()

        mods = []
        for cin, cout in zip(nchanns[:-1], nchanns[1:]):
            mods.append(nn.Conv2d(cin, cout, 3, 1))
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
    pvnet = PVNet(3, 3, [3, 8, 16], 2)
    img = torch.zeros(16, 3, 8, 8)
    print(pvnet(img))
