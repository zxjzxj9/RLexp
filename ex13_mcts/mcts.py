#! /usr/bin/env python

import pyspiel
from utils import SearchNode

class MCTSBot(pyspiel.Bot):
    def __init__(self, game, uct_c, max_simulations):
        self._game = game
        self.max_simulations = max_simulations

    def _apply_tree_policy(self, root, state):
        visit_path = [root]
        working_state = state.clone()
        current_node = root

        while not working_state.is_terminal() and current_node.explore_count > 0:
            if not current_node.children:
                legal_actions = self.evaluator.prior(working_state)
                if current_node is root and self._dirichlet_noise:
                    epsilon, alpha = self._dirichlet_noise
                    noise = self._random_state.dirichlet([alpha] * len(legal_actions))
                    egal_actions = [(a, (1 - epsilon) * p + epsilon * n)
                        for (a, p), n in zip(legal_actions, noise)]
                self._random_state.shuffle(legal_actions)
                player = working_state.current_player()
                current_node.children = [
                     SearchNode(action, player, prior) for action, prior in legal_actions
                ]
                

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
