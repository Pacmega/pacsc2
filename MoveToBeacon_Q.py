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

agent_actions = [SELECT_ARMY, ATTACK_MOVE]

# Agent settings
discount = 0.95         # Discount future rewards

learning_rate = 0.1     # How heavily Q-values are changed based on one experience
learn_decay = 0.0001    # Decay of learning rate
min_learn_rate = 0.005  # Do not decay learning rate below this point

epsilon = 0.9           # Chance of randomly selected action instead of following policy
eps_decay_rate = 0.005  # Decay rate of epsilon
min_epsilon = 0.01      # Do not decay epsilon below this point

# Q-learning settings
field_size = 84 # TODO: this probably shouldn't be hardcoded
pixels_per_bucket = 4
nr_buckets = int(round(field_size / pixels_per_bucket)) # Instead of full size observation space, reduce to this size

def generate_q_table(agent_using_select):
	if agent_using_select:
		components_in_table = 3
		possible_actions = 3		
	else:
		components_in_table = 2
		possible_actions = 2

	discrete_obs_space_size = [nr_buckets] * components_in_table
	q_table = np.zeros(shape=(discrete_obs_space_size + [possible_actions]))
	print("Q table generated with shape {}".format(q_table.shape))
	return q_table

# get_state() returns five values. Only three of those are useful for the Q-table:
#   marine_selected, beacon_x and beacon_y.
# The other two values are useful to calculate a reward for approaching the beacon, 
#   or entirely disregarded if we only reward reaching the beacon itself.
# Coordinates are reduced to fit into the Q-table using nr_buckets.
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


class MoveToBeacon_Q_rich_noselect(base_agent.BaseAgent):
	"""A Q-learning agent that is rewarded both for approaching and for reaching the beacon."""
	# Variables used
	op_every = 10 # Keep a slightly human-like amount of Actions Per Minute
	no_op_counter = 0
	army_selected = False
	selected_action = None

	# Store the previous location in case the API loses the marine's location, tends to happen from time to time.
	q_table = generate_q_table(agent_using_select=False)
	score = 0
	prev_marine_loc = np.zeros(2)

	def reset(self):
		# Runs at the start of every episode
	    super(MoveToBeacon_Q_rich_noselect, self).reset()
	    self.score = 0

	def step(self, obs):
		super(MoveToBeacon_Q_rich_noselect, self).step(obs)
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


class MoveToBeacon_Q_rich(base_agent.BaseAgent):
	"""A Q-learning agent that is rewarded both for approaching and for reaching the beacon."""
	# Variables used
	op_every = 10 # Keep a slightly human-like amount of Actions Per Minute
	no_op_counter = 0
	army_selected = False
	selected_action = None

	# Store the previous location in case the API loses the marine's location, tends to happen from time to time. No idea why.
	prev_marine_loc = np.zeros(2)

	def step(self, obs):
		super(MoveToBeacon_Q, self).step(obs)
		state = get_state(obs.observation)

		if state[1] == 0 and state[2] == 0:
			# API failed to report Marine location, reuse last correct coordinates
			state[1] = prev_marine_loc[0]
			state[2] = prev_marine_loc[1]
		else:
			prev_marine_loc[0] = state[1]
			prev_marine_loc[1] = state[2]

		dist_to_beacon = sqrt(abs(state[1]-state[3])**2 + abs(state[2]-state[4])**2)
		
		print("dist_to_beacon = " + dist_to_beacon)
		print("reward = " + obs.reward)
		return NO_OP


class MoveToBeacon_Q_sparse_noselect(base_agent.BaseAgent):
	"""
	A Q-learning agent that only receives a reward if/when it reaches the beacon.
	Selecting the marine has been taken out of the equation, allowing for quicker results
	"""
	
	# Variables used
	op_every = 10 # Keep a slightly human-like amount of Actions Per Minute
	no_op_counter = 0
	army_selected = False
	selected_action = None

	# Store the previous location in case the API loses the marine's location, tends to happen from time to time. No idea why.
	prev_marine_loc = np.zeros(2)

	def step(self, obs):
		super(MoveToBeacon_Q, self).step(obs)
		new_state = get_state(obs.observation)
		# No need to check if marine_location is okay, not using it
		print("Reminder to check if marine_location is NOT [0, 0], since that messes with reward calculation")
		print("reward = ", obs.reward)
		return NO_OP


class MoveToBeacon_Q_sparse(base_agent.BaseAgent):
	"""A Q-learning agent that only receives a reward if/when it reaches the beacon."""
	
	# Variables used
	op_every = 10 # Keep a slightly human-like amount of Actions Per Minute
	no_op_counter = 0
	army_selected = False
	selected_action = None

	# Store the previous location in case the API loses the marine's location, tends to happen from time to time. No idea why.
	prev_marine_loc = np.zeros(2)

	def step(self, obs):
		super(MoveToBeacon_Q, self).step(obs)
		new_state = get_state(obs.observation)
		# No need to check if marine_location is okay, not using it
		print("Reminder to check if marine_location is NOT [0, 0], since that messes with reward calculation")
		print("reward = ", obs.reward)
		return NO_OP
