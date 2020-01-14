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
