import os
import argparse

# import baselines

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



smart_actions = [ ACTION_DO_NOTHING]

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
        self.total_reward = 0

    def _get_position_of(self, unit_type, unit):
        unit_y, unit_x = (unit_type == unit).nonzero()
        # this works because we only have one unit
        if len(unit_y) == 0 or len(unit_x) == 0:
            return 0,0

        unit_y = int(unit_y.mean() // BLOCK_SIZE)
        unit_x = int(unit_x.mean() // BLOCK_SIZE)

        return unit_y, unit_x

    def get_state(self, obs):
        player_y, player_x = (obs.observation['feature_minimap'][_PLAYER_RELATIVE] == _PLAYER_SELF).nonzero()
        unit_type = obs.observation['feature_screen'][_UNIT_TYPE]

        # -- Update current state
        selected_unit = obs.observation['single_select']
        army_selected = (len(selected_unit) > 0 and
                        selected_unit[0][0] == _ARMY)
        army_y, army_x = self._get_position_of(unit_type, _ARMY)
        #print("Army X:",army_x, " Y:",army_y)

        player_relative = obs.observation["feature_screen"][_PLAYER_RELATIVE]
        beacon_y, beacon_x = self._get_position_of(player_relative, _BEACON)

        #print("Beacon X:",beacon_x, " Y:",beacon_y)

        current_state = [  beacon_x, beacon_y]
        return current_state
    
    def do_action(self, smart_action, obs, army_selected=False):
        smart_action = smart_actions[smart_action]
        print("Selected action: %s" % smart_action)
        
        action = actions.FunctionCall(_NO_OP, [])

        if not army_selected and _SELECT_ARMY in obs.observation['available_actions']:
            print("ACTION_SELECT_ARMY")
            action = actions.FunctionCall(_SELECT_ARMY, [_NOT_QUEUED])
            print("ARMY_SELECTED")

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
                    # print("_MOVE_SCREEN X:",x," Y:",y)
                    action = actions.FunctionCall(_MOVE_SCREEN, [_NOT_QUEUED, [y,x]])
                
        return action
    
    def select_action(self, state, eps=0.5):
        action = self.qlearn.choose_action(str(state))
        return action
    
    def get_reward(self, obs):
        reward = obs.observation['score_cumulative'][0]
        return reward

    def do_learning(self, previous_state, current_state, previous_action, reward):    
        # update learning
        self.qlearn.learn(str(previous_state), previous_action,
                              reward, str(current_state))
      
    
    def step(self, obs):
        current_state = self.get_state(obs)
        if self.previous_action is not None:
            reward = self.get_reward(obs)
            self.do_learning(self.previous_state, current_state, self.previous_action, reward)
        else:
            reward = 0

        selected_action = self.select_action(current_state)    
        action = self.do_action(selected_action, obs)

        # update current state
        self.previous_state = current_state
        self.previous_action = selected_action
        self.total_reward += reward
        return action


class QLearningTable():
    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9):
        self.actions = actions
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy
        self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)
        print("QTABLE: %s" % (self.q_table))

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
        print("State = %s, %s" % (state, type(state)))
        print("Action = %s, %s" % (action, type(action)))

        q_predict = self.q_table.ix[state, action]
        q_target = reward + self.gamma * self.q_table.ix[state_, :].max()

        # update
        self.q_table.ix[state, action] += self.lr * (q_target - q_predict)

    def check_state_exist(self, state):
        if state not in self.q_table.index:
            # Append
            self.q_table = self.q_table.append(pd.Series([0]*len(self.actions), index=self.q_table.columns, name=state))
