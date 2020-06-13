from collections import deque
import os

import numpy as np
import tensorflow as tf
import random
import copy

from peer import Peer
from deepqnetwork import DeepQNetwork
from util import *

class Function(Peer):
    def __init__(self, sim_config, function_config, nn, sess, ID=0, strategy = 'Function'):
        # parameters
        super().__init__(sim_config, ID, strategy)
        self.have_piece = 0
        self.num_history = int(function_config['agent_history'])
        self.nn_object = nn
        self.sess = sess

        self.enable_actions = list([i] for i in range(self.num_peer) if i != self.ID)
        self.enable_actions.insert(0, [])
        
        self.num_channels = int(function_config['num_channels'])
        self.input_data = function_config['input_data']
        if self.input_data == 'upload' or self.input_data == 'download':
            self.width = self.num_history
            self.height = self.num_peer-1
        elif self.input_data == 'upload_and_download':
            self.width = self.num_history
            self.height = (self.num_peer-1)*2

        self.action_selection_input = tf.placeholder(tf.float32, shape = (1, self.height, self.width, self.num_channels))
        self.action_q_values = self.nn_object.q_values(self.action_selection_input)

    def select_random_action(self, neighbor_leecher_list, unchoke_num):
        unchoke_num = self.calculate_unchoke_num(unchoke_num)
        return random.sample(neighbor_leecher_list, unchoke_num)

    def has_enough_history(self):
        return len(self.download_history) > self.num_history

    def select_greedy_action(self, state, neighbor_leecher_list):
        q_values = self.action_q_values.eval(session = self.sess, feed_dict={self.action_selection_input: state})[0]
        q_index = np.argsort(q_values)[::-1]
        for q in q_index:
            if self.enable_actions[q] == []:
                return []
            elif self.enable_actions[q][0] in neighbor_leecher_list:
                return self.enable_actions[q]

    def upload(self, status_list, random_seed = random.randint(0, 1000000)):
        #隣接リーチャーのIDリストを作成
        neighbor_leecher_list = [i for i, s in enumerate(status_list) if i != self.ID and s != 'Seeder']
        #アンチョークする最大人数を決定(例えば隣接リーチャが0の時は誰にもアンチョークできない)
        unchoke_num = min(self.num_unchoke, len(neighbor_leecher_list))
        if self.is_seeder():
            #隣接リーチャの中からランダムに相手を選択.
            return self.seeder_strategy(neighbor_leecher_list, unchoke_num)
        #十分なデータが集まっていない時
        elif not self.has_enough_history():
            self.action_t = self.select_random_action(neighbor_leecher_list, unchoke_num)
            return self.action_t
        else:
            state = self.get_cur_history()
            state = np.reshape(state, [1, self.height, self.width, self.num_channels])
            self.action_t = self.select_greedy_action(state, neighbor_leecher_list)
            return self.action_t

    def get_cur_history(self):
        len_his = len(self.download_history)
        cur_up = copy.deepcopy(self.upload_history[len_his-self.num_history:])
        cur_up = delete_row(cur_up, self.ID)
        cur_down = copy.deepcopy(self.download_history[len_his-self.num_history:])
        cur_down = delete_row(cur_down, self.ID)
        if self.input_data == 'upload':
            return np.transpose(cur_up)
        elif self.input_data == 'download':
            return np.transpose(cur_down)
        elif self.input_data == 'upload_and_download':
            up_down = np.r_[cur_up, cur_down]
            up_down = up_down.transpose()
            if np.shape(up_down) == (self.num_peer-1, self.num_history*2):
                #print('ok ',np.shape(up_down))
                up_down = up_down.reshape((self.num_peer-1)*2, self.num_history)
            else:
                #print('no ', np.shape(up_down))
                up_down = None
            return up_down 

