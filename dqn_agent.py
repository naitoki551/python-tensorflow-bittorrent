from collections import deque
import os

import numpy as np
import tensorflow as tf
import random
import copy

from peer import Peer
from deepqnetwork import DeepQNetwork
from util import *

class DQNAgent(Peer):
    def __init__(self, sim_config, agent_config, ID = 0, strategy = 'Agent', training_flag= True):
        # parameters
        super().__init__(sim_config, ID, strategy)
        self.num_history = agent_config['agent_history']
        self.training_flag = training_flag
        ## エージェントの行動の集合
        self.enable_actions = list([i] for i in range(self.num_peer) if i != self.ID)
        self.enable_actions.insert(0, [])

        self.num_actions = self.num_peer

        ## 各パラメータの値を格納(詳細はtraining_config.pyに記載)
        self.minibatch_size = int(agent_config['minibatch_size'])
        self.learning_rate = float(agent_config['learning_rate'])
        self.discount_factor = float(agent_config['discount_factor'])
        self.max_D_size = int(agent_config['max_D_size'])
        self.min_D_size = int(agent_config['min_D_size'])
        self.network_update_frequency = int(agent_config['network_update_frequency'])
        self.epsilon_decaying_states = int(agent_config['epsilon_decaying_states'])
        self.min_epsilon = float(agent_config['min_epsilon'])
        self.reward_config = str(agent_config['reward_config'])
        self.momentum = float(agent_config['momentum'])
        self.opt_epsilon = float(agent_config['opt_epsilon'])

        # replay memory
        self.D = deque(maxlen=self.max_D_size)
        # variables
        self.current_loss = 0.0
        self.current_Q_max = 0.0
        self.num_total_states = 0
        self.Q_max = 0
        self.action_t = []
        self.reward_t = [0]
        self.action_t_past = 0

        # model
        self.graph = tf.Graph()
        with self.graph.as_default():
            #input_layer
            self.input_data = agent_config['input_data']
            if self.input_data == 'upload' or self.input_data == 'download':
                self.width = self.num_history
                self.height = self.num_peer-1
            elif self.input_data == 'upload_and_download':
                self.width = self.num_history
                self.height = (self.num_peer-1)*2

            self.num_channels = int(agent_config['num_channels'])
            # train_network
            self.tf_train_input = tf.placeholder(tf.float32, shape=(self.minibatch_size, self.height, self.width, self.num_channels))
            self.tf_train_target = tf.placeholder(tf.float32, shape=(self.minibatch_size, self.num_actions))
            self.tf_filter_input = tf.placeholder(tf.float32, shape=(self.minibatch_size, self.num_actions))
            self.train_network = DeepQNetwork(self.width, self.height, self.num_actions, agent_config)
            ###
            #self.train_q_values = self.train_network.q_values(self.tf_train_input)

            # target_network
            self.tf_target_input = tf.placeholder(tf.float32, shape=(self.minibatch_size, self.height, self.width, self.num_channels))
            #self.tf_target_input = tf.placeholder(tf.float32, shape=(1, self.height, self.width, self.num_channels))
            #self.tf_target_input = tf.placeholder(tf.float32, [self.minibatch_size, width, height])
            self.target_network = DeepQNetwork(self.width, self.height, self.num_actions, agent_config)
            self.target_q_values = self.target_network.q_values(self.tf_target_input)

            # アクション選択用のプレースホルダー
            self.tf_action_selection_input = tf.placeholder(tf.float32, shape=(1, self.height, self.width,  self.num_channels))
            self.action_q_values = self.train_network.q_values(self.tf_action_selection_input)


            # loss function
            self.loss = self.train_network.clipped_loss(self.tf_train_input, self.tf_train_target, self.tf_filter_input)

            # optimizer
            self.optimizer = tf.train.RMSPropOptimizer(self.learning_rate, momentum=self.momentum, epsilon=self.opt_epsilon, name='RMSProp')
            self.training = self.optimizer.minimize(self.loss, name='training')
            
            self.sess = tf.InteractiveSession()
            self.sess.run(tf.global_variables_initializer())
            self.update_target_network()

    def update_target_network(self):
        self.train_network.copy_network_to(self.target_network, self.sess)

    def select_random_action(self, neighbor_leecher_list, unchoke_num):
        unchoke_num = self.calculate_unchoke_num(unchoke_num)
        return random.sample(neighbor_leecher_list, unchoke_num)

    def get_q_max(self):
        if self.num_history > len(self.download_history):
            return 0
        else:
            state = self.get_cur_history()
            state = np.reshape(state, [1, self.height, self.width, self.num_channels])
            q_values = self.action_q_values.eval(session = self.sess, feed_dict={self.tf_action_selection_input: state})[0]
            return np.max(q_values)

    def select_greedy_action(self, state, neighbor_leecher_list):
        q_values = self.action_q_values.eval(session = self.sess,feed_dict={self.tf_action_selection_input: state})[0]
        q_index = np.argsort(q_values)[::-1]
        for q in q_index:
            if self.enable_actions[q] == []:
                return []
            elif self.enable_actions[q][0] in neighbor_leecher_list:
                return self.enable_actions[q]
        #self.action_t = self.enable_actions[q_index[0]]
        # action_t = []
        # for i in q_index:
        #     if i in neighbor_leecher_list:
        #         action_t.append(self.enable_actions[i])
        #         if len(action_t) == self.num_unchoke:
        #             return action_t
        #     elif i == self.ID:
        #         [action_t.append(self.enable_actions[i])
        #         return action_t
    
    def upload(self, status_list):
        self.enable_actions = list([i] for i in range(self.num_actions) if i != self.ID)
        self.enable_actions.insert(0, [])
        #print(self.enable_actions)
        epsilon = self.calculate_epsilon()
        neighbor_leecher_list = [i for i in range(len(status_list)) if status_list[i] != 'Seeder' and i != self.ID]
        self.unchoke_probability = 1-(1/(len(neighbor_leecher_list)+1))
        unchoke_num = min(self.num_unchoke, len(neighbor_leecher_list))
        ### 隣接リーチャがいない時
        if neighbor_leecher_list == []:
            self.action_t = []
            return self.action_t
        ### 十分なデータが集まっていない時
        elif len(self.download_history) <= self.num_history or not self.has_enough_memory():
            self.action_t = self.select_random_action(neighbor_leecher_list, unchoke_num)
            return self.action_t
        else:
            ### イプシロンの確率でランダム戦略
            if random.random() <= epsilon:
                self.action_t = self.select_random_action(neighbor_leecher_list, unchoke_num)
                return self.action_t
            else:
                ### 直近の履歴情報を返す
                state = self.get_cur_history()
                state = np.reshape(state, [1, self.height, self.width, self.num_channels])
                ### Q値によって選択
                self.action_t = self.select_greedy_action(state, neighbor_leecher_list)
                return self.action_t

    def store_experience(self, state, action, reward, state_1, terminal):
        self.D.append((state, action, reward, state_1, terminal))

    def experience_replay(self):
        state_minibatch = []
        action_minibatch = []
        reward_minibatch = []
        state_1_minibatch = []
        terminal_minibatch = []

        # sample random minibatch
        minibatch_size = min(len(self.D), self.minibatch_size)
        minibatch_indexes = np.random.randint(0, len(self.D), minibatch_size)

        for j in minibatch_indexes:
            state_j, action_j, reward_j, state_j_1, terminal = self.D[j]
            action_j_index = action_j
            state_minibatch.append(state_j)
            action_minibatch.append(action_j)
            reward_minibatch.append(reward_j)
            state_1_minibatch.append(state_j_1)
            terminal_minibatch.append(terminal)

        state_minibatch = np.reshape(state_minibatch, [self.minibatch_size, self.height, self.width, self.num_channels])
        state_1_minibatch = np.reshape(state_1_minibatch, [self.minibatch_size, self.height, self.width, self.num_channels])
        ### target_networkでQ値を算出(教師データ)
        ### train_q_valuesはtarget_networkを使って学習を行わない(2013年版DQN)
        #target_qs = self.train_q_values.eval(feed_dict={self.tf_train_input: state_1_minibatch})
        target_qs = self.target_q_values.eval(session = self.sess, feed_dict={self.tf_target_input: state_1_minibatch})
        target = np.zeros(shape=(self.minibatch_size, self.num_actions), dtype=np.float32)
        q_value_filter = np.zeros(shape=(self.minibatch_size, self.num_actions), dtype=np.float32)
        
        for i in range(self.minibatch_size):
            terminal = terminal_minibatch[i]
            action_index = action_minibatch[i]
            #print(action_index)
            reward = reward_minibatch[i]
            target[i][action_index] = reward if terminal else reward + self.learning_rate * np.max(target_qs[i])
            q_value_filter[i][action_index] = 1.0

        _, self.current_loss = self.sess.run([self.training, self.loss], feed_dict={self.tf_train_input: state_minibatch,
                                    self.tf_train_target: target,
                                    self.tf_filter_input: q_value_filter})

    def load_model(self, file_path):
        self.train_network.restore_parameters(self.sess, file_path)

    ### CNNのモデルを保存
    def save_model(self, save_path, num_episode):
        if not os.path.exists(os.path.join(save_path, 'model','train_network')):
            os.makedirs(os.path.join(save_path, 'model','train_network'))
        if not os.path.exists(os.path.join(save_path, 'model','target_network')):
            os.makedirs(os.path.join(save_path, 'model','target_network'))
        self.train_network.save_parameters(self.sess, save_path+'/model/train_network/train_network', num_episode)
        self.target_network.save_parameters(self.sess, save_path+'/model/target_network/train_network', num_episode)

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

    def calculate_epsilon(self):
        if self.epsilon_decaying_states == 0:
            return self.min_epsilon
        else:
            return max(self.min_epsilon, 1.0-(self.num_total_states/self.epsilon_decaying_states))

    def has_enough_memory(self):
        return len(self.D) >= self.min_D_size

    def next_step(self):
        if self.num_history < len(self.download_history) and self.training_flag:
            self.num_total_states += 1
            #時刻tの状態, 行動, 報酬
            state_t = self.state_t_past
            action_t = self.action_t_past
            state_t_1 = self.get_cur_history()
            #現在の状態を保存しておき, 次のステップで用いる
            self.state_t_past = state_t_1
            self.action_t_past = self.action_t

            self.reward_t = []
            if self.reward_config == 'all_download':
                self.reward_t.append(sum(self.current_download))
            elif self.reward_config == 'each_download':
                for a_t in action_t:
                    self.reward_t.append(self.current_download[a_t])
            elif self.reward_config == 'each_download_penalty':
                for a_t in action_t:
                    each_download = self.current_download[a_t]
                    if a_t == self.ID:
                        self.reward_t.append(0)
                    elif each_download >= 1:
                        self.reward_t.append(1)
                    else:
                        self.reward_t.append(-1)
            reward_t = self.reward_t

            #状態を更新
            super().next_step()
            terminal = self.is_seeder()

            ## 行動とアクションを再生メモリに格納
            self.store_experience(state_t, [self.enable_actions.index(action_t)], reward_t, state_t_1, terminal)
            ## 十分なメモリが揃ったら学習を開始
            if self.has_enough_memory():
                if (self.num_total_states%self.network_update_frequency)==0:
                    print('-----update_network------')
                    self.update_target_network()
                self.experience_replay()
        else:
            self.state_t_past = self.get_cur_history()
            self.action_t_past = self.action_t
            super().next_step()


    ### オブジェクトをリセットしないため, このメソッドを呼び出して次のエピソードへ移行する. 
    def reset_history(self):
        #持っているピース.デフォルトは0
        self.have_piece = 0
        #隣接ピアへのアップロード履歴を保存する変数(多分0か1しか検討しなさそうやけど)
        self.current_upload = [0 for i in range(self.num_peer)]	
        #隣接ピアからの直近のダウンロード履歴を保存する変数
        #初期はすべて1を入れてランダムに行うようにする(0入れて一番古いのを1にするのもあり)
        self.current_download = [0 for i in range(self.num_peer)]
        
        #隣接ピアからのダウンロード履歴を保存する変数
        self.upload_history = [[0 for i in range(self.num_peer)]]
        #隣接ピアからのアップロード履歴を保存する変数
        self.download_history = [[1 for i in range(self.num_peer)]]
        self.strategy = 'Agent'
