from pysc2.agents import base_agent
from pysc2.lib import actions
from pysc2.lib import features

import numpy as np

_PLAYER_SELF = features.PlayerRelative.SELF
_PLAYER_NEUTRAL = features.PlayerRelative.NEUTRAL  # The beacon falls under Neutral team

FUNCTIONS = actions.FUNCTIONS

# Agent settings
discount = 0.95         # Discount future rewards

learning_rate = 0.1     # How heavily Q-values are changed based on one experience
learn_decay = 0.0001    # Decay of learning rate
min_learn_rate = 0.005  # Do not decay learning rate below this point

epsilon = 0.9           # Chance of randomly selected action instead of following policy
eps_decay_rate = 0.005  # Decay rate of epsilon
min_epsilon = 0.01      # Do not decay epsilon below this point

qtable_size = 20 # Instead of 84x84 observation space, reduce to this dimension

# State is a tuple: (selected_unit (ID), marine_loc_x, marine_loc_y, beacon_centre_loc_x, beacon_centre_loc_y)
def get_state(observation):
	"""
	Returns a simplified version of the game state:
	marine_selected - 0 or 1 (False/True)
	marine_location_x - int between 0 and qtable_size
	marine_location_y - int between 0 and qtable_size
	beacon_centre_location_x - int between 0 and qtable_size
	beacon_centre_location_y - int between 0 and qtable_size
	"""
	marine_sel = 0 # False, except as int (to represent everything as int)
	if FUNCTIONS.Attack_screen.id in observation.available_actions:
		# _PLAYER_SELF has one of their own units selected
		marine_sel = 1 # True, except as int (to represent everything as int)

	player_relative = observation.feature_screen.player_relative
	marine_location = np.mean(_xy_locs(player_relative == _PLAYER_SELF), axis=0).round()
	m_loc_reduced = tuple(coordinate / qtable_size for coordinate in marine_location)
	beacon_centre_location = np.mean(_xy_locs(player_relative == _PLAYER_NEUTRAL), axis=0).round()
	b_cen_reduced = tuple(coordinate / qtable_size for coordinate in beacon_centre_location)

	return (marine_sel, *(m_loc_reduced), *(b_cen_reduced))

# Functional actions for this environment
NO_OP = FUNCTIONS.no_op()
SELECT_ARMY = FUNCTIONS.select_army('select')
def ATTACK_MOVE(target):
 return FUNCTIONS.Attack_screen("now", target)

# Copied from scripted_agent.py as found in sc2py/agents
def _xy_locs(mask):
	"""Mask should be a set of bools from comparison with a feature layer."""
	y, x = mask.nonzero()
	return list(zip(x, y))

class MoveToBeacon_hardcoded(base_agent.BaseAgent):
	# Variables used
	op_every = 10 # Keep a slightly human-like amount of Actions Per Minute
	no_op_counter = 0
	army_selected = False
	selected_action = None

	def step(self, obs):
		super(MoveToBeacon_hardcoded, self).step(obs)
		print(get_state(obs.observation))
		if self.no_op_counter == self.op_every:
			if FUNCTIONS.Attack_screen.id in obs.observation.available_actions:
				# Army units selected
				player_relative = obs.observation.feature_screen.player_relative
				beacon_centre_location = np.mean(_xy_locs(player_relative == _PLAYER_NEUTRAL), axis=0).round()
				self.selected_action = ATTACK_MOVE(beacon_centre_location)
			else:
				self.selected_action = SELECT_ARMY
			self.no_op_counter = 0
			return self.selected_action
		else:
			self.no_op_counter += 1
			return NO_OP
