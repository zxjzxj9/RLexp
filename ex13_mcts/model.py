#! /usr/bin/env python

import torch
import torch.nn as nn


class PVNet(nn.Module):
    def __init__(self, in_chan, mid_chan, out_chan):
        super().__init__()

        self.featnet = None

        self.policy = None
        self.value = None

    def forward(self, x, mask):
        pass

