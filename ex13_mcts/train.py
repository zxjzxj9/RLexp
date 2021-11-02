#! /usr/bin/env python

import pyspiel
import numpy as np

game = pyspiel.load_game("tic_tac_toe")
state = game.new_initial_state()
# print(state)
# print(game.num_distinct_actions())

