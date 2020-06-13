import numpy as np
import os
import csv
import tensorflow as tf
from deepqnetwork import DeepQNetwork
import os


#与えられた2次元配列を指定の行だけ消す. 返すのはリスト型
def delete_row(lst2d, row):
    lst2d = np.delete(lst2d, row, 1)
    lst2d = lst2d.tolist()
    return lst2d

### Function用のagent_config.csvの読み込みを行う関数
def create_function_nn(sim_config):
    function_config = dict()
    function_path = sim_config['Function_path']
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), function_path)

    with open(path + '/agent_config.csv', 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            function_config[row[0]] = row[1]
    
    function_config['min_epsilon'] = 0
    function_config['epsilon_decaying_states'] = 0
    function_config['min_D_size'] = 0  

    input_data = function_config['input_data']
    num_history = int(function_config['agent_history'])
    num_peer = int(sim_config['num_peer'])
    if input_data == 'upload' or input_data == 'download':
        width = num_history
        height = num_peer-1
    elif input_data == 'upload_and_download':
        width = num_history
        height = (num_peer-1)*2

    function_config['width'] = width
    function_config['height'] = height
    neural_network = DeepQNetwork(width, height, int(function_config['fc2_outputs']), function_config)
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    neural_network.restore_parameters(sess, os.path.join(path, 'model', 'train_network', 'train_network-'+str(sim_config['Function_restore_checkpoint'])))

    return neural_network, sess, function_config


### トレーニング用の保存ファイルを作成
def make_training_file(file_name):
    #指定の環境名のところまでのパス
    file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../training_data', file_name)
    
    #ディレクトリ番号が書かれたファイルを指すパス
    file_num_path = os.path.join(file_path,'.file_index')
    
    #同一環境名のパスが存在していない場合
    #初期ディレクトリ0を作成してそこに保存
    if os.path.exists(file_num_path) == False:
        file_index = 0
    else:
        with open(file_num_path, 'r') as f:
            file_index = int(f.read()) + 1
    
    #ディレクトリを作成
    os.makedirs(os.path.join(file_path, str(file_index)))
    with open(file_num_path, 'w') as f:
        f.write(str(file_index))

    return os.path.join(file_path, str(file_index))


### テスト用の保存ファイルを作成
def make_test_file(file_name):
    #指定の環境名のところまでのパス
    file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../test_data', file_name)
    
    #ディレクトリ番号が書かれたファイルを指すパス
    file_num_path = os.path.join(file_path,'.file_index')
    
    #同一環境名のパスが存在していない場合
    #初期ディレクトリ0を作成してそこに保存
    if os.path.exists(file_num_path) == False:
        file_index = 0
    else:
        with open(file_num_path, 'r') as f:
            file_index = int(f.read()) + 1
    
    #ディレクトリを作成
    os.makedirs(os.path.join(file_path, str(file_index)))
    with open(file_num_path, 'w') as f:
        f.write(str(file_index))

    return os.path.join(file_path, str(file_index))



# 文字列を適切な型に変換する
def convert_type(s):
    try: return int(s)
    except:
        try: return float(s)
        except: return str(s)
