import numpy as np
import copy
import random

class Peer:
    def __init__(self, sim_config, ID, strategy):

        self.num_peer = sim_config['num_peer']	
        #ピアID
        self.ID = ID
        #戦略(ランダム戦略やしっぺ返し戦略など)
        self.strategy = strategy
        #アンチョークする確率(アンチョーク可能な時に敢えて確率的にアンチョークする上限を制限する)
        #もし, この値が0.7ならば30%の確率でアンチョークしない
        self.unchoke_probability = 1
        #シーダーに移行するピース数
        self.num_piece = sim_config['num_piece'] 
        #最大アンチョーク人数
        self.num_unchoke = sim_config['num_unchoke']
        
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
        self.download_history = [[0 for i in range(self.num_peer)]]
        
        self.first_action = True

    def upload(self):
        pass #オーバーライド
    
    def is_seeder(self):
        return self.strategy == 'Seeder' or self.strategy == 'Dummy'

    def update_upload(self, to_id, up_bw):
        self.current_upload[to_id] = up_bw

    def update_download(self, from_id, up_bw):
        self.current_download[from_id] = up_bw
        self.have_piece += up_bw

    def next_step(self):
        self.upload_history.append(copy.deepcopy(self.current_upload))
        self.download_history.append(copy.deepcopy(self.current_download))
        self.current_upload = [0 for i in range(self.num_peer)]
        self.current_download = [0 for i in range(self.num_peer)]
        if self.have_piece >= self.num_piece:
            self.strategy = 'Seeder'

    #確率的にアンチョークする人数を算出する関数
    def calculate_unchoke_num(self, num_unchoke):
        num = 0
        for i in range(num_unchoke):
            if random.random() <= self.unchoke_probability:
                num += 1
        return num

    def get_strategy(self):
        return self.strategy

    ### シーダー戦略 ###
    #シーダーはアンチョークを絞らない
    def seeder_strategy(self, neighbor_leechers, num):
        return random.sample(neighbor_leechers, num)

    #インスタンスをリセットしないため, このメソッドを呼び出して次のエピソードへ移行する. 
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
        self.strategy = self.__class__.__name__
    

