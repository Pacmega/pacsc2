# TODO: document that I would've like to implement an agent for a much more sparse reward environment,
#   but didn't have time to implement the systems that might make that possible

from pysc2.agents import base_agent
from pysc2.lib import actions
from pysc2.lib import features

from math import sqrt

import numpy as np

_PLAYER_SELF = features.PlayerRelative.SELF
_PLAYER_NEUTRAL = features.PlayerRelative.NEUTRAL  # The beacon falls under Neutral team

FUNCTIONS = actions.FUNCTIONS

# Functional actions for this environment
NO_OP = FUNCTIONS.no_op()
SELECT_ARMY = FUNCTIONS.select_army('select')
def ATTACK_MOVE(target):
 return FUNCTIONS.Attack_screen("now", target)

# Action NONE is only for the first time, to make sure no reward is given. Can't be selected.
action_list = ['SELECT_ARMY', "MOVE_UP", "MOVE_DOWN", "MOVE_LEFT", "MOVE_RIGHT",
				"MOVE_UPLEFT", "MOVE_UPRIGHT", "MOVE_DOWNLEFT", "MOVE_DOWNRIGHT", "NONE"]

# Agent settings
discount = 0.95         # Discount future rewards

learning_rate = 0.1     # How heavily Q-values are changed based on one experience
learn_decay = 0.001    # Decay of learning rate
min_learn_rate = 0.005  # Do not decay learning rate below this point

epsilon = 0.9           # Chance of randomly selected action instead of following policy
eps_decay_rate = 0.005  # Decay rate of epsilon
min_epsilon = 0.01      # Do not decay epsilon below this point

# Q-learning settings
field_size = 84 # TODO: this probably shouldn't be hardcoded
pixels_per_bucket = 8
nr_buckets = int(round(field_size / pixels_per_bucket)) # Instead of full size observation space, reduce to this size
components_in_table = 5 # Unit selected, Marine X & Y, Beacon X & Y

def generate_q_table(diagonal_move_enabled=False):

	# possible actions: select army + either 4 directional or 8 directional movement
	if diagonal_move_enabled:
		possible_actions = 9
	else:
		possible_actions = 5

	discrete_obs_space_size = [nr_buckets] * components_in_table
	q_table = np.zeros(shape=(discrete_obs_space_size + [possible_actions]))
	print("Q table generated with shape {}".format(q_table.shape))
	return q_table

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


class MoveToBeacon_Q_nondiag(base_agent.BaseAgent):
	"""A Q-learning agent that is rewarded both for approaching and for reaching the beacon."""
	# Variables used
	op_every = 10 # Keep a slightly human-like amount of Actions Per Minute
	no_op_counter = 0
	
	q_table = generate_q_table(diagonal_move_enabled=False)
	score = 0
	prev_action = action_list.index("NONE")
	
	# Store the previous location in case the API loses the marine's location,
	#   tends to happen from time to time when the marine touches the beacon.
	prev_marine_loc = np.zeros(2)

	# def calc_reward(self, reward):
		

	def reset(self):
		# Runs at the start of every episode
	    super(MoveToBeacon_Q_nondiag, self).reset()
	    self.score = 0
	    self.prev_action = action_list.index("NONE")
	    prev_marine_loc = np.zeros(2)

	def step(self, obs):
		super(MoveToBeacon_Q_nondiag, self).step(obs)
		# TODO: WARNING - Reward is reward for PREVIOUS action, so obs.reward is for last action
		#   Addendum: prev_marine_loc instead of previous state to update Q table with
		self.state = get_state(obs.observation)

		if self.state[1] == 0 and self.state[2] == 0:
			# API failed to report Marine location, reuse last correct coordinates
			self.state[1] = self.prev_marine_loc[0]
			self.state[2] = self.prev_marine_loc[1]
		else:
			self.prev_marine_loc[0] = self.state[1]
			self.prev_marine_loc[1] = self.state[2]

		self.dist_to_beacon = sqrt(abs(self.state[1]-self.state[3])**2 + abs(self.state[2]-self.state[4])**2)
		
		# print("dist_to_beacon = " + str(self.dist_to_beacon))
		# print("reward = " + obs.reward)
		return NO_OP
