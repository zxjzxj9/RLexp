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

    def puct_value(self, parent_explore_count, uct_c):
        if self.outcome is not None:
          return self.outcome[self.player]
    
        return ((self.explore_count and self.total_reward / self.explore_count) +
                uct_c * self.prior * math.sqrt(parent_explore_count) /
                (self.explore_count + 1))

    def sort_key(self):
        return (0 if self.outcome is None else self.outcome[self.player],
            self.explore_count, self.total_reward)

    def best_child(self):
        return max(self.children, key=SearchNode.sort_key)

    def children_str(self, state=None):
        return "\n".join([
            c.to_str(state)
            for c in reversed(sorted(self.children, key=SearchNode.sort_key))
        ])

    def to_str(self, state=None):
        action = (
            state.action_to_string(state.current_player(), self.action)
            if state and self.action is not None else str(self.action))
        return ("{:>6}: player: {}, prior: {:5.3f}, value: {:6.3f}, sims: {:5d}, "
            "outcome: {}, {:3d} children").format(
            action, self.player, self.prior, self.explore_count and
            self.total_reward / self.explore_count, self.explore_count,
            ("{:4.1f}".format(self.outcome[self.player])
                if self.outcome else "none"), len(self.children))

    def __str__(self):
        return self.to_str(None)
