from pysc2.agents import base_agent
from pysc2.lib import actions
from pysc2.lib import features

import numpy as np

_PLAYER_SELF = features.PlayerRelative.SELF
_PLAYER_NEUTRAL = features.PlayerRelative.NEUTRAL  # The beacon falls under Neutral team

FUNCTIONS = actions.FUNCTIONS

# Functional actions for this environment
NO_OP = FUNCTIONS.no_op()
SELECT_ARMY = FUNCTIONS.select_army('select')
def ATTACK_MOVE(target):
	return FUNCTIONS.Attack_screen("now", target)

# Q-learning settings
field_size = 84 # TODO: this probably shouldn't be hardcoded
pixels_per_bucket = 4
nr_buckets = int(round(field_size / pixels_per_bucket)) # Instead of full size observation space, reduce to this size

def get_state(observation):
	"""
	Returns a simplified version of the game state:
	marine_selected - 0 or 1 (False/True)
	marine_location_x - int between 0 and nr_buckets
	marine_location_y - int between 0 and nr_buckets
	beacon_centre_location_x - int between 0 and nr_buckets
	beacon_centre_location_y - int between 0 and nr_buckets
	"""
	marine_sel = 0 # False, except as int (to represent everything as int)

	if FUNCTIONS.Attack_screen.id in observation.available_actions:
		# _PLAYER_SELF has one of their own units selected
		marine_sel = 1 # True, except as int (to represent everything as int)

	player_relative = observation.feature_screen.player_relative
	marine_location = np.mean(_xy_locs(player_relative == _PLAYER_SELF), axis=0)

	if type(marine_location) is not np.ndarray:
		# API messed up and didn't properly return the marine's location. 
		# This only seems to happen when the marine reaches the target, but is not consistent?
		#  This is fine for action selection since attack-move only requires a target, 
		#   but will break reward calculation if not handled outside of this function.
		marine_location = np.zeros(2)

	m_loc_reduced = tuple(int(round(coordinate / nr_buckets)) for coordinate in marine_location)
	beacon_centre_location = np.mean(_xy_locs(player_relative == _PLAYER_NEUTRAL), axis=0)
	b_cen_reduced = tuple(int(round(coordinate / nr_buckets)) for coordinate in beacon_centre_location)

	return (marine_sel, *(m_loc_reduced), *(b_cen_reduced))


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
		# TODO: WARNING - Reward is reward for PREVIOUS action, so obs.reward is for last action
		print(get_state(obs.observation))
		print(obs.reward)
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
