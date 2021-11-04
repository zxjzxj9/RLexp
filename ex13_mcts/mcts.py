#! /usr/bin/env python

import pyspiel
from utils import SearchNode

class MCTSBot(pyspiel.Bot):
    def __init__(self, game, uct_c, max_simulations):
        self._game = game

    def mcts_search(self, state):
        root_player = state.current_player()
        root = SearchNode(None, state.current_player(), 1)
        for _ in range(self.max_simulations):
            visit_path, working_state = self._apply_tree_policy(root, state)
            if working_state.is_terminal():
                returns = working_state.returns()
                visit_path[-1].outcome = returns
                solved = self.solve
            else:
                returns = self.evaluator.evaluate(working_state)
                solved = False
