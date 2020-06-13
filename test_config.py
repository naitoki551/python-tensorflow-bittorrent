from collections import OrderedDict
import numpy as np

## ここでシミュレーションの設定をおこなう．
def test_config():

    ############################################################################################
    ### trainingで獲得された戦略を復元させるために，戦略が格納されているファイル名を指定する ###
    ############################################################################################
    file_config = OrderedDict()
    ## テストする戦略が格納されたファイルパスを指定(traing_data下のファイル)
    ## ((ファイル名1)，(ファイル名2)，restore_checkpoint(復元するエピソード数))
    ## 例えば，../training_data/example1/ex1の5000エピソード目の戦略をテストしたい場合，('example1', 'ex1', 5000)をリストに追加する．
    ## 複数個の戦略を同時にテストすることも可能で，リストに追加していけば良い
    file_config['restore_file'] = [('example1', 'ex1', 5000), ('example1', 'ex1', 2500)]
    
    ## テストの結果を保存するファイル名．
    ## 例えば，'example_test'と設定すれば，../test_data/example_test/(数字)直下ににそのデータが格納される
    file_config['save_file'] = 'examle_test'

    
    ##########################################################
    ### テストシミュレーション全体に関するパラメータの設定 ###
    ##########################################################
    sim_config = OrderedDict()
    sim_config['iteration'] = 100    #シミュレーション回数
    sim_config['num_peer'] = 21         #ピア数
    sim_config['num_piece'] = 500       #ピアがシーダになるピース数
    sim_config['num_unchoke'] = 1       #ピアが選択できるアンチョークの数 

    # 比較する戦略
    sim_config['compare_strategy'] = ['TFTOU','TFT', 'Random']

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



    return sim_config, file_config

