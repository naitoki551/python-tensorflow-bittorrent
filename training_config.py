from collections import OrderedDict
import numpy as np

## ここでシミュレーションの設定をおこなう．
def training_config():

    ################################################
    ### trainingデータを保存するファイル名の設定 ###
    ################################################
    file_config = OrderedDict()
    ## trainingデータは'./training_data/(save_file)/(index)'下に保存される(indexは自動的に割り当てられる)
    file_config['file_name'] = 'example1'



    ####################################################
    ### シミュレーション全体に関するパラメータの設定 ###
    ####################################################
    sim_config = OrderedDict()
    sim_config['num_episode'] = 20000    #エピソード数
    sim_config['num_peer'] = 21         #ピア数
    sim_config['num_piece'] = 500       #ピアがシーダになるピース数
    sim_config['num_unchoke'] = 1       #ピアが選択できるアンチョークの数 

    ### 隣接ピアの戦略に関する設定
    sim_config['TFT_history'] = 3   #TFTOUまたはTFTのピアが参照する履歴数

    ## Function戦略のCNNモデルが保存されているファイル名と復元させるチェックポイント
    sim_config['Function_path'] = '../training_data/example1/ex1'
    sim_config['Function_restore_checkpoint'] = 5000

    ### 隣接ピアが取る戦略- ['(戦略名)'，(この戦略を取るピア数)]
    ##   Seeder - ランダムにアンチョーク先を決定. ダウンロードを行わない
    ##   Function - DQNによって獲得された戦略を取る隣接ピア．
    ##   TFTRandom - ダウンロード量が多い順にアンチョーク先を決定. 足りない分は無作為にアンチョーク先を決定する
    ##   TFT - ダウンロード量が多い順にアンチョーク先を決定. 足りない分はアンチョークを行わない
    ##   Random - 無作為にアンチョーク先を決定
    sim_config['neighbor_strategy'] = np.array([
                                            ['Seeder', 0],
                                            ['Function', 2],
                                            ['TFTOU', 8],
                                            ['TFT', 0],
                                            ['Random', 10]
        ])
    
    ### 戦略の振り分け方法
    ##   ORDER_ID - ピアのID順に戦略を昇順に振り分け
    ##   RANDOM_ID - 戦略を取る人数は指定, ピアIDは無作為に振り分け
    ##   RANDOM_ALL - ピアのIDと戦略を取る人数が無作為に振り分け．ただしピア数が0であればその戦略は取らない
    sim_config['strategy_allocation'] = 'RANDOM_ID'  
    
    ### 何エピソード毎にCNNのモデルやデータを保存するか．
    sim_config['save_point'] = 100




    ################################
    ### エージェントに関する設定 ###
    ################################
    agent_config = OrderedDict()
    agent_config['agent_strategy'] = 'DQNAgent'  #エージェント(学習者)の戦略．これと同じクラス名を持つ戦略がエージェントの戦略となる．
    agent_config['agent_history'] = 10     #エージェント(学習者)が参照する履歴数
        
    #dqnの構成
    agent_config['minibatch_size'] = 32
    agent_config['learning_rate'] = 0.00025
    agent_config['discount_factor'] = 0.99
        
    ## replay memory Dの最小サイズ(min_D_size)最大サイズ(max_D_size)
    ## 学習はmin_D_size回以上の行動選択(=経験の数)を経なければ学習を行わない
    ## max_D_size以上の経験数を経るとDから最も古い経験を消去する
    agent_config['min_D_size'] = 20000 #defalt
    #agent_config['min_D_size'] = 50
    agent_config['max_D_size'] = 100000

    ## CNNの重みを更新する頻度
    agent_config['network_update_frequency'] = 10000

    ## epsilonを1.0からmin epsilonの値まで減少させるまでの行動選択の回数
    ## オプション: epsilon_decaying_states=0にしたら常にmin_epsilonに固定される
    agent_config['epsilon_decaying_states'] = 500000
    #agent_config['epsilon_decaying_states'] = 0
    agent_config['min_epsilon'] = 0.1
        
    ## モーメンタム
    agent_config['momentum'] = 0.95
        
    ## RMSPropoptimizerのイプシロン
    agent_config['opt_epsilon'] = 1e-2
    #agent_config['opt_epsilon'] = 1e-10 # defalt
        
    ### rewardをどう定義するか．以下オプションの記述．
    ##      all_download : 全てのピアから受けたダウンロードの総和(0 <= r_t <= num_peer)
    ##      each_download : アップロードしたピアが次の時刻で返してくれたダウンロード量(0 <= r_t <= 1)
    ##      each_download_penalty : アップロードしたピアが次の時刻で返してくれたダウンロード量．ただし，返されなかった場合-1とする(-1 <= r_t <= 1)
    agent_config['reward_config'] = 'all_download'
    #agent_config['reward_config'] = 'each_download'
    #agent_config['reward_config'] = 'each_download_penalty'

    ### 入力データとして与えるデータ．以下オプションの記述
    ##      upload_and_download : アップロード履歴とダウンロード履歴の両方
    ##      upload : 隣接ピアのアップロード履歴のみ
    ##      download : 隣接ピアのダウンロード履歴のみ
    agent_config['input_data'] = 'upload_and_download'
    #agent_config['input_data'] = 'upload'
    #agent_config['input_data'] = 'download'
        

    ### 畳み込み層1の構造
    #チャネルの数
    agent_config['num_channels'] = 1
    #フィルターの数
    agent_config['conv1_filter_num'] = 8
    #フィルターのサイズ
    agent_config['conv1_filter_size_x'] = 6
    agent_config['conv1_filter_size_y'] = 2
        
    #フィルターのストライド
    agent_config['conv1_stride_x'] = 1
    agent_config['conv1_stride_y'] = 2

    ### 畳み込み層2の構造
    #フィルターの数
    agent_config['conv2_filter_num'] = 16

    #フィルターのサイズ
    agent_config['conv2_filter_size_x'] = 3
    agent_config['conv2_filter_size_y'] = 1
    #フィルターのストライド
    agent_config['conv2_stride_x'] = 1
    agent_config['conv2_stride_y'] = 1

    #全結合層の構成
    agent_config['fc1_outputs'] = 512
    agent_config['fc2_outputs'] = sim_config['num_peer']

    return sim_config, agent_config, file_config

