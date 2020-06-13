import numpy as np
import random
import copy
import csv
import time
import matplotlib.pyplot as plt
from collections import OrderedDict
from scipy import stats

from dqn_agent import DQNAgent
from peer_strategy.function_st import Function
from peer_strategy.tftou import TFTOU
from peer_strategy.tft import TFT
from peer_strategy.random_st import Random
#from peer_strategy.function_nn import *
from test_config import *
from util import *

class TestSimuration:
    def __init__(self, sim_config, file_config):
        self.iteration = sim_config['iteration']
        self.num_peer = sim_config['num_peer']
        self.num_piece = sim_config['num_piece']
        self.num_unchoke = sim_config['num_unchoke']
        self.neighbor_strategy = sim_config['neighbor_strategy']
        self.strategy_allocation = sim_config['strategy_allocation'] 
        self.save_point = sim_config['save_point']

        self.sim_config = sim_config
        self.file_config = file_config

        ## 観測者のオブジェクトを格納するためのリスト
        self.observer_object_list = []
        self.compare_strategy = sim_config['compare_strategy']

        ### Function戦略のためのCNNを構成する
        tmp = self.neighbor_strategy[np.any(self.neighbor_strategy == 'Function', axis=1)]
        if int(tmp[0][1]) > 0:
            self.function_nn, self.function_sess, self.function_config  = create_function_nn(sim_config)

        ### シミュレーションの設定ファイルを保存
        ## (make_training_file関数はutil.pyにプログラムを記述)
        self.path = make_test_file(file_config['save_file'])
        del sim_config['neighbor_strategy']
        with open(self.path + '/sim_config.csv', 'a') as f:
            writer = csv.writer(f, lineterminator='\n')
            for k ,v in sim_config.items():
                writer.writerow([k,v])
        with open(self.path + '/neighbor_strategy.csv', 'w') as f:
            writer = csv.writer(f, lineterminator='\n')
            writer.writerow(['隣接ピアが取る戦略', '人数'])
            writer.writerows(self.neighbor_strategy)


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

    ##########################################
    ### 観測者のオブジェクトを作成する関数 ###
    ##########################################
    def create_observer(self):
        for i, r_path in enumerate(self.file_config['restore_file']):
            path = os.path.join('../training_data',r_path[0], r_path[1])
            _agent_config = dict()
            with open(path + '/agent_config.csv', 'r') as f:
                reader = csv.reader(f)
                for row in reader:
                    _agent_config[row[0]] = convert_type(row[1])
            _agent_config['min_epsilon'] = 0
            _agent_config['epsilon_decaying_states'] = 0
            _agent_config['min_D_size'] = 0
            _agent_object = (globals()[_agent_config['agent_strategy']](self.sim_config, _agent_config))
            _agent_object.load_model(os.path.join(path, 'model', 'train_network', 'train_network-' + str(r_path[2])))
            _agent_object.training_flag = False  # 学習を行わないように
            self.observer_object_list.append(_agent_object)
        for st in self.compare_strategy:
            self.observer_object_list.append(globals()[st](self.sim_config))
 

    ####################################
    ### シミュレーションを動かす関数 ###
    ####################################
    def run_sim(self):
        ### シミュレーションデータを格納するための辞書
        sim_data = OrderedDict()
        sim_data['iteration'] = []
        sim_data['download_time'] = []
        sim_data['download_per_step'] = []
        sim_data['upload_per_step'] = []
        ## 処理上，一時的にデータを格納するための辞書
        temp_data = OrderedDict()
        for k in sim_data.keys():
            temp_data[k] = []
        
        ## 観測者となる戦略を全てインスタンス化する
        self.create_observer()

        ## ループの開始
        for iter in range(1, self.iteration+1):
            print('ITERATIONS : ', iter)
            sim_data['iteration'].append([iter])
            for ob_index, ob_object in enumerate(self.observer_object_list):
                ### ランダムシードを指定して，比較する戦略(観測者の取る戦略)毎に隣接ピアの戦略が変化しないようにする
                random.seed(iter)
                peer_strategy_list = self.allocate_strategy()
                peer_object_list = self.create_peer(ob_object, peer_strategy_list)
                step = 0
                terminal = False
                while not terminal:
                    step += 1

                    ### 各ピアがアップロード先を選択する
                    status_list = ['Seeder' if p.is_seeder() else 'Leecher' for p in peer_object_list]  # 各ピアがシーダか，リーチャかを格納するリスト
                    choose = dict() #各ピアの選択を格納する辞書型
                    for i, p in enumerate(peer_object_list):
                        random.seed(iter + step*10000 + i*23)
                        choose[p.ID] = p.upload(status_list)

                    ### 各ピアのアップロード・ダウンロードを反映させる
                    for from_id, to_ids in zip(choose.keys(), choose.values()):
                        for to_id in to_ids:
                            self.exchange_pieces(peer_object_list[from_id], peer_object_list[to_id])
                    
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

                    ## 観測者がシーダになったら1回のエピソードが終了する
                    terminal = ob_object.is_seeder()
                    
                    ## 10000ステップ後にエージェントがシーダになっていない場合，エピソードを強制終了する
                    ## (ほとんど使わないが，念の為記述．戦略によっては無限ループになるかもしれないので)
                    if step > 10000:
                        print('--------- Forcibly end an episode ---------')
                        terminal = True
                    
                    if terminal:
                        temp_data['download_time'].append(step)
                        temp_data['download_per_step'].append(self.num_piece/step)
                        temp_data['upload_per_step'].append(np.sum(ob_object.upload_history)/step)

            ## 観測者の履歴をリセットする
            for ob_object in self.observer_object_list:
                ob_object.reset_history()
            for k, v in temp_data.items():
                sim_data[k].append(v)
                temp_data[k] = []
        
        ##########################################
        ### シミュレーションのデータを出力する ###
        ##########################################
        def save_data():
            label = ['平均値','最大値', '75%点', '中央値', '25%点', '最小値', '分散']
            header = []
            for f in self.file_config['restore_file']:
                header.append(f[0]+'/'+ f[1] +'('+ str(f[2]) + ')')
            for st in self.compare_strategy:
                header.append(st)
            write_data = dict()
            for k, v in sim_data.items():
                if k != 'iteration':
                    write_data[k] = [
                                        np.average(v, axis=0).tolist(),
                                        np.max(v, axis=0).tolist(),
                                        stats.scoreatpercentile(v, 75, axis=0).tolist(),
                                        np.median(v, axis=0).tolist(),
                                        stats.scoreatpercentile(v, 25, axis=0).tolist(),
                                        np.min(v, axis=0).tolist(),
                                        np.var(v, axis=0).tolist()
                    ]
                    for i, lb in enumerate(label):
                        write_data[k][i].insert(0, lb)
            
            ### 数値データをresult.csvに出力
            with open(self.path + '/result.csv', 'a') as f:
                writer = csv.writer(f, lineterminator='\n')
                for k in sim_data.keys():
                    if k != 'iteration':
                        writer.writerow([k]+header)
                        writer.writerows(write_data[k])
                        writer.writerow([''])
            
            ###グラフを出力
            def save_fig(fig_name, data):
                plt.rcParams['font.family'] = 'IPAPGothic'
                fig, ax = plt.subplots()
                bp = ax.boxplot(tuple(np.transpose(data)))
                ax.set_xticklabels(header)
                plt.title(fig_name)
                plt.grid()
                plt.savefig(self.path + '/figure/' + fig_name+'.png')
    
            ## ディレクトリを作成
            os.makedirs(os.path.join(self.path, 'figure'))
            for k, v in sim_data.items():
                if k != 'iteration':
                    save_fig(k, v)
                
        save_data()



if __name__ == '__main__':
    print('   --------------------------------------')
    print('  | The TEST simulation will start soon. |')
    print('  | Please wait a moment.                |')
    print('   --------------------------------------')
    sim_config, file_config = test_config()

    sim = TestSimuration(sim_config, file_config)
    sim.run_sim()