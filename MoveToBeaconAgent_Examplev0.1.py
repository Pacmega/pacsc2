import os
import argparse

import baselines

from pysc2.lib import actions as sc2_actions
from pysc2.env import environment
from pysc2.lib import features
from pysc2.lib import actions
from pysc2.agents import base_agent

import pandas as pd
import numpy as np

_PLAYER_SELF = 1
_PLAYER_RELATIVE = features.SCREEN_FEATURES.player_relative.index

_UNIT_TYPE = features.SCREEN_FEATURES.unit_type.index
_PLAYER_ID = features.SCREEN_FEATURES.player_id.index
_ARMY = 48
_BEACON = 3  # beacon/minerals
# ACTIONS
_NO_OP = actions.FUNCTIONS.no_op.id
_SELECT_ARMY = actions.FUNCTIONS.select_army.id
_MOVE_SCREEN = actions.FUNCTIONS.Move_minimap.id

_NOT_QUEUED = [0]
_SELECT_ALL = [0]

ACTION_DO_NOTHING = 'donothing'
ACTION_SELECT_ARMY = 'selectarmy'
ACTION_MOVE_SCREEN = 'movescreen'



smart_actions = [ ACTION_DO_NOTHING, ACTION_SELECT_ARMY]

import re
pattern = ACTION_MOVE_SCREEN + '_(\d+)_(\d+)'

# Create actions to move over the screen
BLOCK_SIZE = 16
for mm_x in range(int(BLOCK_SIZE/2), 64, BLOCK_SIZE):
    for mm_y in range(int(BLOCK_SIZE/2), 64, BLOCK_SIZE):
        smart_actions.append(ACTION_MOVE_SCREEN + '_' + str(mm_x) + '_' + str(mm_y))
        print("ACTION_MOVE_SCREEN_",str(mm_x),"_",str(mm_y))

class MoveToBeaconAgent(base_agent.BaseAgent):
    def __init__(self):
        super(MoveToBeaconAgent, self).__init__()

        self.qlearn = QLearningTable(actions=list(range(len(smart_actions))))

        self.previous_action = None
        self.previous_state = None
        self.previousArmyX = 0
        self.previousArmyY = 0

    def _get_position_of(self, unit_type, unit):
        unit_y, unit_x = (unit_type == unit).nonzero()
        # this works because we only have one unit
        if len(unit_y) == 0 or len(unit_x) == 0:
            return 0,0

        unit_y = int(unit_y.mean() / BLOCK_SIZE)
        unit_x = int(unit_x.mean() / BLOCK_SIZE)

        return unit_y, unit_x

    def step(self, obs):

        player_y, player_x = (obs.observation['minimap'][_PLAYER_RELATIVE] == _PLAYER_SELF).nonzero()
        unit_type = obs.observation['screen'][_UNIT_TYPE]

        # -- Update current state
        selected_unit = obs.observation['single_select']
        army_selected = (len(selected_unit) > 0 and
                        selected_unit[0][0] == _ARMY)
        army_y, army_x = self._get_position_of(unit_type, _ARMY)
        print("Army X:",army_x, " Y:",army_y)

        player_relative = obs.observation["screen"][_PLAYER_RELATIVE]
        beacon_y, beacon_x = self._get_position_of(player_relative, _BEACON)

        print("Beacon X:",beacon_x, " Y:",beacon_y)

        current_state = [  army_selected, beacon_x, beacon_y]

        # -- Learn
        if self.previous_action is not None:
            reward = obs.observation['score_cumulative'][0]

            print("reward: ",reward)

            self.qlearn.learn(str(self.previous_state), self.previous_action,
                              reward, str(current_state))

        # -- Chose new action
        rl_action = self.qlearn.choose_action(str(current_state))
        smart_action = smart_actions[rl_action]

        # update previous state
        self.previous_state = current_state
        self.previous_action = rl_action

        self.previousArmyX = army_x
        self.previousArmyY = army_y

        action = actions.FunctionCall(_NO_OP, [])

        if smart_action == ACTION_DO_NOTHING:
            print("ACTION_DO_NOTHING")
            action = actions.FunctionCall(_NO_OP,[])
        elif smart_action == ACTION_SELECT_ARMY:
            if _SELECT_ARMY in obs.observation['available_actions']:
                print("ACTION_SELECT_ARMY")
                action = actions.FunctionCall(_SELECT_ARMY, [_NOT_QUEUED])
        elif ACTION_MOVE_SCREEN in smart_action:
            if _MOVE_SCREEN in obs.observation['available_actions']:
                match = re.match(pattern, smart_action)
                if match is not None:
                    x, y = int(match.group(1)), int(match.group(2))
                    print("_MOVE_SCREEN X:",x," Y:",y)
                    action = actions.FunctionCall(_MOVE_SCREEN, [_NOT_QUEUED, [y,x]])

        print("----------------------------")
        return action


class QLearningTable():
    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9):
        self.actions = actions
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy
        self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)

    def choose_action(self, observation):
        self.check_state_exist(observation)

        if np.random.uniform() < self.epsilon:
            # exploitation
            state_action = self.q_table.ix[observation, :]
            state_action = state_action.reindex(np.random.permutation(state_action.index))
            action = state_action.idxmax()
        else:
            # exploration
            action = np.random.choice(self.actions)
        return action

    def learn(self, state, action, reward, state_):
        self.check_state_exist(state_)
        self.check_state_exist(state)

        q_predict = self.q_table.ix[state, action]
        q_target = reward + self.gamma * self.q_table.ix[state_, :].max()

        # update
        self.q_table.ix[state, action] += self.lr * (q_target - q_predict)

    def check_state_exist(self, state):
        if state not in self.q_table.index:
            # Append
            self.q_table = self.q_table.append(pd.Series([0]*len(self.actions), index=self.q_table.columns, name=state))
