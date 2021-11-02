#! /usr/bin/env python

class SearchNode(object):
    __slots__ = [
        "action",
        "player",
        "prior",
        "explore_count",
        "total_reward",
        "outcome",
        "children",
    ]

    def __init__(self, action, player, prior):
        self.action = action
        self.player = player
        self.prior = prior
        self.explore_count = 0
        self.total_reward = 0.0
        self.outcome = None
        self.children = []

  def uct_value(self, parent_explore_count, uct_c):
      if self.outcome is not None:
          return self.outcome[self.player]

      if self.explore_count == 0:
          return float("inf")

      return self.total_reward / self.explore_count + uct_c * math.sqrt(
          math.log(parent_explore_count) / self.explore_count)
