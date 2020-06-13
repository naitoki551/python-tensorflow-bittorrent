import numpy as np
import random
import copy
import csv
import time
import matplotlib.pyplot as plt
from collections import OrderedDict


from dqn_agent import DQNAgent
from peer_strategy.function_st import Function
from peer_strategy.tftou import TFTOU
from peer_strategy.tft import TFT
from peer_strategy.random_st import Random
#from peer_strategy.function_nn import *
from training_config import *
from util import *

class TraingSimuration:
    def __init__(self, sim_config, agent_config, file_config):
        self.num_episode = sim_config['num_episode']
        self.num_peer = sim_config['num_peer']
        self.num_piece = sim_config['num_piece']
        self.num_unchoke = sim_config['num_unchoke']
        self.neighbor_strategy = sim_config['neighbor_strategy']
        self.strategy_allocation = sim_config['strategy_allocation'] 
        self.save_point = sim_config['save_point']

        self.sim_config = sim_config

        ### エージェントのインスタンス化
        agent_class = (globals()[agent_config['agent_strategy']])
        self.agent_object = agent_class(sim_config, agent_config)

        ### シミュレーションの設定ファイルを保存
        ## (make_training_file関数はutil.pyにプログラムを記述)
        self.path = make_training_file(file_config['file_name'])
        del sim_config['neighbor_strategy']
        with open(self.path + '/sim_config.csv', 'a') as f:
            writer = csv.writer(f, lineterminator='\n')
            for k ,v in sim_config.items():
                writer.writerow([k,v])
        with open(self.path + '/neighbor_strategy.csv', 'a') as f:
            writer = csv.writer(f, lineterminator='\n')
            writer.writerow(['隣接ピアが取る戦略', '人数'])
            writer.writerows(self.neighbor_strategy)
        with open(self.path + '/agent_config.csv', 'a') as f:
            writer = csv.writer(f, lineterminator='\n')
            for k ,v in agent_config.items():
                writer.writerow([k,v])

        ### Function戦略のためのCNNを構成する
        tmp = self.neighbor_strategy[np.any(self.neighbor_strategy == 'Function', axis=1)]
        if int(tmp[0][1]) > 0:
            self.function_nn, self.function_sess, self.function_config  = create_function_nn(sim_config)
        
            ## Function戦略のconfigを保存
            # with open(self.path + '/function.csv', 'w') as f:
            #     writer = csv.writer(f, lineterminator='\n')
            #     for k ,v in self.function_config.items():
            #         writer.writerow([k,v])


    ######################################
    ### ピアの戦略の割り振りを行う関数 ###
    ######################################
    def allocate_strategy(self):
        _ns = copy.deepcopy(self.neighbor_strategy)
        _ns = _ns[np.all(_ns!='0', axis=1)]

        peer_strategy_list = list()

        #ピアのID順に戦略を昇順に振り分け
        if self.strategy_allocation == 'ORDER_ID':
            for e in _ns:
                for _ in range(int(e[1])):
                    peer_strategy_list.append(e[0])
            peer_strategy_list.insert(0, 'Agent')

        #人数は指定, ピアIDは無作為に振り分け
        elif self.strategy_allocation == 'RANDOM_ID':
            for e in _ns:
                for _ in range(int(e[1])):
                    peer_strategy_list.append(e[0])
            peer_strategy_list.insert(0, 'Agent')
            random.shuffle(peer_strategy_list)

        #ピアのID, 戦略ごとの人数を無作為に戦略を振り分け(ただし, 人数が1以上のもののみ)
        elif self.strategy_allocation == 'RANDOM_ALL':
            peer_strategy_list = [np.random.choice(_ns) for _ in range(self.num_peer-1)]
            peer_strategy_list.insert(random.randint(0, self.num_peer-1), 'Agent')
        
        else:
            print('The value of \'strategy_allocation\' is not appropriate.')
            exit()

        return np.array(peer_strategy_list)

    ######################################
    ### ピアのインスタンス化を行う関数 ###
    ######################################
    def create_peer(self, agent_object, peer_strategy_list):
        peer_object_list = []
        for i, ps in enumerate(peer_strategy_list):
            if 'Agent' in ps:
                agent_object.ID = i
                peer_object_list.append(agent_object)
            else:
                peer_class = (globals()[ps])
                if ps == 'Function':
                    peer_object_list.append(peer_class(self.sim_config, self.function_config, self.function_nn, self.function_sess, i, ps))
                else:
                    peer_object_list.append(peer_class(self.sim_config, i, ps))

        return peer_object_list

    ##############################################
    ### ピア間のピースのやり取りを反映する関数 ###
    ##############################################
    def exchange_pieces(self, sent_peer, received_peer, up_bw=1):
        #print(sent_peer.ID, ' -> (',up_bw, ')-> ',received_peer.ID)
        if sent_peer != received_peer:
            sent_peer.update_upload(received_peer.ID, up_bw)
            received_peer.update_download(sent_peer.ID, up_bw)

    ####################################
    ### シミュレーションを動かす関数 ###
    ####################################
    def run_sim(self):
        ### シミュレーションデータを格納するための辞書
        sim_data = OrderedDict()
        sim_data['episode'] = []
        sim_data['epsilon'] = []
        sim_data['download_time'] = []
        sim_data['loss'] = []
        sim_data['Q_max'] = []
        sim_data['reward'] = []
        sim_data['download_per_step'] = []
        sim_data['upload_per_step'] = []
        ## 処理上，一時的にデータを格納するための辞書
        temp_data = OrderedDict()
        for k in sim_data.keys():
            temp_data[k] = []

        ## Agentにnum_episodeの数だけエピソードを経験させる．
        for ep in range(1, self.num_episode+1): 
            start_time = time.time()
            ### 隣接ピアに戦略を割り振り，ピアのインスタンス化を行う．
            ## peer_strategy_listはピアの戦略名が格納されたリスト．
            ## peer_object_listはピアのオブジェクトが格納されているリスト．
            peer_strategy_list = self.allocate_strategy()
            peer_object_list = self.create_peer(self.agent_object, peer_strategy_list)

            ### 変数の準備
            step = 0
            terminal = False    # エージェントがシーダになるとTrueになる(つまりエピソードの終了を表す)
            self.agent_object.reset_history()
            
            ### 学習の状況を可視化するための変数
            loss = 0.0
            Q_max = 0.0
            reward = 0

            ### 1回分のエピソードの開始 
            while not terminal:
                step += 1

                ### 各ピアがアップロード先を選択する
                status_list = ['Seeder' if p.is_seeder() else 'Leecher' for p in peer_object_list]  # 各ピアがシーダか，リーチャかを格納するリスト
                choose = dict() #各ピアの選択を格納する辞書型
                for p in peer_object_list:
                    choose[p.ID] = p.upload(status_list)

                ### 各ピアのアップロード・ダウンロードを反映させる
                for from_id, to_ids in zip(choose.keys(), choose.values()):
                    for to_id in to_ids:
                        self.exchange_pieces(peer_object_list[from_id], peer_object_list[to_id])

                Q_max += self.agent_object.get_q_max()
                loss += self.agent_object.current_loss
                reward += sum(self.agent_object.reward_t)

                ### 次のエピソードの準備
                for p in peer_object_list:
                    p.next_step()

                ### ステップ毎にデータを標準出力：
                ### 消したい場合は'print_info()'の記述をコメントアウトしてください．
                def print_info():
                    print("-----------------------------------------------------------------------------------------------------\nt = %d" % step)
                    for p in peer_object_list:
                        print('ID: %d(%-10s)\tpiece: %-5d\tup: %d %-8s\tdown: %d %s'% (p.ID, p.strategy, p.have_piece, sum(p.upload_history[step]), choose[p.ID], sum(p.download_history[step]), [i for i,d in enumerate(p.download_history[step]) if d >= 1]))
                #print_info()
                
                ## エージェントがシーダになったら1回のエピソードが終了する
                terminal = self.agent_object.is_seeder()

                ## 10000ステップ後にエージェントがシーダになっていない場合，エピソードを強制終了する
                ## (ほとんど使わないが，念の為記述．戦略によっては無限ループになるかもしれないので)
                if step > 10000:
                    print('--------- Forcibly end an episode ---------')
                    break

            ###################################
            ### 1回のエピソード終了後の処理 ###
            ###################################
            print('==============================================================================================')
            print('EPISODE: {:06d}/{:06d} | EPSILON: {:.4f} |LOSS: {:.4f} | Q MAX: {:.4f} | NEIGHBORS (Function - {:d}, TFTRandom - {:d}, TFT-{:d}, Random{:d})| DOWNLOAD TIME: {:03d} | DOWNLOAD PER STEP: {:.4f} | EXEC TIME : {:.3f} SEC'.format(
                    ep, 
                    self.num_episode, 
                    self.agent_object.calculate_epsilon(), 
                    loss / (step-self.agent_object.num_history), 
                    Q_max / (step-self.agent_object.num_history), 
                    int(np.sum(peer_strategy_list == 'Function')),
                    int(np.sum(peer_strategy_list == 'TFTOU')),
                    int(np.sum(peer_strategy_list == 'TFT')),
                    int(np.sum(peer_strategy_list == 'Random')),
                    step, 
                    self.num_piece/step,
                    time.time() - start_time
                ))

            ### シミュレーションのデータを一時的に保存
            temp_data['epsilon'].append(self.agent_object.calculate_epsilon())
            temp_data['download_time'].append(step)
            temp_data['loss'].append(loss/(step-self.agent_object.num_history))
            temp_data['Q_max'].append(Q_max/step-self.agent_object.num_history)
            temp_data['reward'].append(reward/step)
            temp_data['download_per_step'].append(self.num_piece/step)
            temp_data['upload_per_step'].append(np.sum(self.agent_object.upload_history)/step)
            
            ### (save_point)episodeごとに各データ(temp_data)の値の平均値をsim_dataに格納して記録しておく
            if ep%self.save_point == 0:
                sim_data['episode'].append(ep)
                for k, v in temp_data.items():
                    if k != 'episode':
                        sim_data[k].append(np.mean(v))
                        temp_data[k] = []
                ## CNNの重みも保存しておく
                self.agent_object.save_model(self.path, ep)


        ##########################################
        ### シミュレーションのデータを出力する ###
        ##########################################
        def save_data():
            ### 数値データをresult.csvに出力
            with open(self.path + '/result.csv', 'a') as f:
                keys = list(sim_data.keys())
                values = np.transpose(list(sim_data.values()))
                writer = csv.writer(f)
                writer.writerow(keys)
                writer.writerows(values)
                       
            
            ### 図示化する
            ## clfは図の初期化. 前回の設定が残らないように
            #グラフを出力
            def save_fig(x, y, fig_name, y_lim=None):
                plt.clf()                    
                plt.plot(x, y, linewidth=0.5)
                plt.title(fig_name)
                plt.xlabel('episode')
                plt.ylabel(fig_name)
                if y_lim:
                    plt.ylim(y_lim)
                plt.savefig(os.path.join(self.path, 'figure', 'fig_'+fig_name))
            ## ディレクトリを作成
            os.makedirs(os.path.join(self.path, 'figure'))
            for key, value in sim_data.items():
                if key == 'episode':
                    continue
                elif key == 'epsilon':
                    save_fig(sim_data['episode'], value, key, y_lim=[0,1.1])
                else:
                    save_fig(sim_data['episode'], value, key)

        save_data()




if __name__ == '__main__':
    print('   ------------------------------------------')
    print('  | The TRAINNING simulation will start soon. |')
    print('  | Please wait a moment.                    |')
    print('   ------------------------------------------')
    sim_config, agent_config, file_config = training_config()

    sim = TraingSimuration(sim_config, agent_config, file_config)
    sim.run_sim()