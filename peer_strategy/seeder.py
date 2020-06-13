import numpy as np
import random
import itertools

from peer import Peer

class Random(Peer):
    def __init__(self, sim_config, ID=0, strategy = 'Seeder'):
        super().__init__(sim_config, ID, strategy)
        self.have_piece = sim_config['num_pieces']

    def upload(self, status_list, random_seed = random.randint(0, 1000000)):
        #random.seed(random_seed)
        #隣接リーチャーのIDリストを作成
        neighbor_leecher_list = [i for i, s in enumerate(status_list) if i != self.ID and s != 'Seeder']
        self.unchoke_probability = 1-(1/(len(neighbor_leecher_list)+1))
        #アンチョークする最大人数を決定(例えば隣接リーチャが0の時は誰にもアンチョークできない)
        unchoke_num = min(self.num_unchoke, len(neighbor_leecher_list))

        return self.seeder_strategy(neighbor_leecher_list, unchoke_num)
