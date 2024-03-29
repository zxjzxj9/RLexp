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
                    legal_actions = [(a, (1 - epsilon) * p + epsilon * n)
                        for (a, p), n in zip(legal_actions, noise)]
                self._random_state.shuffle(legal_actions)
                player = working_state.current_player()
                current_node.children = [
                     SearchNode(action, player, prior) for action, prior in legal_actions
                ]
            if working_state.is_chance_node():
                outcomes = working_state.chance_outcomes()
                action_list, prob_list = zip(*outcomes)
                action = self._random_state.choice(action_list, p=prob_list)
                chosen_child = next(c for c in current_node.children if c.action == action)
            else:
                chosen_child = max(
                    current_node.children,
                    key=lambda c: self._child_selection_fn(
                    c, current_node.explore_count, self.uct_c))
            working_state.apply_action(chosen_child.action)
            current_node = chosen_child
            visit_path.append(current_node)
        return visit_path, working_state

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

            for node in reversed(visit_path):
                node.total_reward += returns[root_player if node.player ==
                    pyspiel.PlayerId.CHANCE else node.player]
                node.explore_count += 1

            if solved and node.children:
                player = node.children[0].player
                if player == pyspiel.PlayerId.CHANCE:
                    outcome = node.children[0].outcome
                    if (outcome is not None and
                        all(np.array_equal(c.outcome, outcome) for c in node.children)):
                        node.outcome = outcome
                    else:
                        solved = False
                else:
                    best = None
                    all_solved = True

                    for child in node.children:
                        if child.outcome is None:
                            all_solved = False
                        elif best is None or child.outcome[player] > best.outcome[player]:
                            best = child
                    if (best is not None and
                        (all_solved or best.outcome[player] == self.max_utility)):
                        node.outcome = best.outcome
                    else:
                        solved = False

            if root.outcome is not None:
                break
        return root
