import numpy as np
import random
import itertools

from peer import Peer

class TFTOU(Peer):
    def __init__(self, sim_config, ID=0, strategy = 'TFTOU'):
        super().__init__(sim_config, ID, strategy)
        self.num_history = sim_config['TFT_history']
    

    def tft_random_strategy(self, neighbor_leecher_list, unchoke_num):
        #参照する履歴数を決定する
        h_num = min(self.num_history, len(self.download_history))
        
        #このピアが参照するダウンロード履歴をseen_download_historyに格納
        seen_download_history = np.array(self.download_history[len(self.download_history)-h_num:])
        
        #各ピアごとのダウンロードの合計値を算出
        sum_download = sum(seen_download_history)

        #確率的にアンチョークする人数を決定する
        unchoke_num = self.calculate_unchoke_num(unchoke_num)

        choose = []
        #ref は基準値. アップロードが多い順に取っていく. 0(ダウンロードしていないピア)もとる
        for ref in range(max(sum_download), -1, -1):
            #基準値に一致するピアのIDをlstに格納
            lst = [l for l in neighbor_leecher_list if sum_download[l] == ref]
            #現在選ばれているピアがunchoke人数より多い場合, unchoke人数までのピアをlstからランダムに選択して格納
            if len(choose)+len(lst) >= unchoke_num:
                choose.extend(random.sample(lst, unchoke_num-len(choose)))
                break
            #確実に選ばれるピアをchooseに格納
            choose.extend(lst)
        return choose

    def upload(self, status_list, random_seed = random.randint(0, 1000000)):
        #random.seed(random_seed)
        #隣接リーチャーのIDリストを作成
        neighbor_leecher_list = [i for i, s in enumerate(status_list) if i != self.ID and s != 'Seeder']
        #アンチョークする最大人数を決定(例えば隣接リーチャが0の時は誰にもアンチョークできない)
        unchoke_num = min(self.num_unchoke, len(neighbor_leecher_list))

        if self.is_seeder():
            #隣接リーチャの中からランダムに相手を選択.
            return self.seeder_strategy(neighbor_leecher_list, unchoke_num)
        else:
            return self.tft_random_strategy(neighbor_leecher_list, unchoke_num)