import numpy as np
import tensorflow as tf
from deepqnetwork import DeepQNetwork
import os

def create_function_nn(sim_config):
    function_config = dict()
    function_path = sim_config['Function_path']
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), function_path)

    with open(path + '/agent_config.csv', 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            funcion_config[row[0]] = row[1]
    
    func_config['min_epsilon'] = 0
    func_config['epsilon_decaying_states'] = 0
    func_config['min_D_size'] = 0  

    input_data = function_config['input_data']
    num_history = int(function_config['agent_history'])
    num_peer = int(sim_config['num_peer'])
    if input_data == 'upload' or input_data == 'download':
        width = num_history
        height = num_peer-1
    elif input_data == 'upload_and_download':
        width = num_history
        height = (num_peer-1)*2

    func_config['width'] = width
    func_config['height'] = height
    neural_network = DeepQNetwork(width, height, int(func_config['fc2_outputs']), function_config)
    sess = tf.InteractiveSession()
    sess.run(tf.global_variabels_initializer())
    neural_network.restore_parameters(sess, os.path.join(path, 'model', 'train_network', 'train_network-'+str(sim_config['Function_restore_checkpoint'])))

    return neural_network, sess, func_config