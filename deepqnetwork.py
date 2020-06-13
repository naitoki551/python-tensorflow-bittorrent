
import tensorflow as tf

class DeepQNetwork:
    def __init__(self, width, height, num_actions, agent_config):
        #conv1
        self.num_channels = int(agent_config['num_channels'])
        self.conv1_filter_size_x = int(agent_config['conv1_filter_size_x'])
        self.conv1_filter_size_y = int(agent_config['conv1_filter_size_y'])
        self.conv1_filter_num = int(agent_config['conv1_filter_num'])
        self.conv1_stride_x = int(agent_config['conv1_stride_x'])
        self.conv1_stride_y = int(agent_config['conv1_stride_y'])
        self.conv1_output_size_x = int((width - self.conv1_filter_size_x) / self.conv1_stride_x + 1)
        self.conv1_output_size_y = int((height - self.conv1_filter_size_y) / self.conv1_stride_y + 1)

        self.conv1_weights, self.conv1_biases = self.create_conv_net([self.conv1_filter_size_y, self.conv1_filter_size_x, self.num_channels, self.conv1_filter_num], name='conv1')

        # print('conv1_weights: ', self.conv1_weights.get_shape().as_list())
        # print(width, '*', height, '*', self.num_channels)
        # print((width - self.conv1_filter_size_x) / self.conv1_stride_x + 1)
        # print('->\t', self.conv1_output_size_x)
        # print(((height - self.conv1_filter_size_y) / self.conv1_stride_y + 1))
        # print('->\t', self.conv1_output_size_y)
        
        #conv2
        self.conv2_filter_size_x = int(agent_config['conv2_filter_size_x'])
        self.conv2_filter_size_y = int(agent_config['conv2_filter_size_y'])
        self.conv2_stride_x = int(agent_config['conv2_stride_x'])
        self.conv2_stride_y = int(agent_config['conv2_stride_y'])
        self.conv2_output_size_x = int((self.conv1_output_size_x - self.conv2_filter_size_x) / self.conv2_stride_x + 1)
        self.conv2_output_size_y = int((self.conv1_output_size_y - self.conv2_filter_size_y) / self.conv2_stride_y + 1)

        ## 畳み込み層にする場合
        self.conv2_filter_num = int(agent_config['conv2_filter_num'])
        self.conv2_weights, self.conv2_biases = self.create_conv_net([self.conv2_filter_size_y, self.conv2_filter_size_x, self.conv1_filter_num, self.conv2_filter_num], name='conv2')

        # print((self.conv1_output_size_x - self.conv2_filter_size_x) / self.conv2_stride_x + 1)
        # print('->\t', self.conv2_output_size_x)
        # print((self.conv1_output_size_y - self.conv2_filter_size_y) / self.conv2_stride_y + 1)
        # print('->\t', self.conv2_output_size_y)

        ## 畳み込み層の場合
        self.fc1_inputs = self.conv2_output_size_x * self.conv2_output_size_y * self.conv2_filter_num
        
        self.fc1_outputs = int(agent_config['fc1_outputs'])
        self.fc1_weights, self.fc1_biases = self.create_fc_net([self.fc1_inputs, self.fc1_outputs], name='fc1')

        # fc2
        self.fc2_inputs = self.fc1_outputs
        self.fc2_outputs = num_actions
        self.fc2_weights, self.fc2_biases = self.create_fc_net([self.fc2_inputs, self.fc2_outputs], name='fc2')

        # Network variable saver
        self.saver = tf.train.Saver({var.name: var for var in self.weights_and_biases()},max_to_keep=None)


    def forward(self, data):
        conv1 = tf.nn.conv2d(data, self.conv1_weights, strides=[1, self.conv1_stride_y, self.conv1_stride_x, 1], padding='VALID')
        conv1_cutoff = tf.nn.relu(conv1 + self.conv1_biases)
        
        # print('conv1', conv1.get_shape().as_list())
        # print('conv1_cutoff', conv1_cutoff.get_shape().as_list())

        conv2 = tf.nn.conv2d(conv1_cutoff, self.conv2_weights,[1, self.conv2_stride_y, self.conv2_stride_x,1], padding='VALID') 
        conv2_cutoff = tf.nn.relu(conv2 + self.conv2_biases)
        # print('conv2',conv2.get_shape().as_list())
        # print('conv2_cutoff',conv2_cutoff.get_shape().as_list())

        shape = conv2.get_shape().as_list()
        reshape = tf.reshape(conv2_cutoff, [shape[0], shape[1] * shape[2] * shape[3]])
        # print(shape)
        # print(shape[1] * shape[2] * shape[3])

        
        fc1 = tf.nn.relu(tf.matmul(reshape, self.fc1_weights) + self.fc1_biases)
        fc2 = tf.matmul(fc1, self.fc2_weights) + self.fc2_biases
        return fc2

    def q_values(self, data):
        return self.forward(data)

    def filtered_q_values(self, data, q_value_filter):
        return tf.multiply(self.q_values(data), q_value_filter)

    def loss(self, data, target, q_value_filter):
        filtered_qs = self.filtered_q_values(data, q_value_filter)
        return tf.reduce_mean(tf.nn.l2_loss(target - filtered_qs))


    def clipped_loss(self, data, target, q_value_filter):
        filtered_qs = self.filtered_q_values(data, q_value_filter)
        error = tf.abs(target - filtered_qs)
        quadratic = tf.clip_by_value(error, 0.0, 1.0)
        linear = error - quadratic
        return tf.reduce_mean(0.5 * tf.square(quadratic) + linear)
    
    def create_conv_net(self, shape, name):
        weights = tf.Variable(tf.truncated_normal(shape=shape, stddev=0.01), name=name + 'weights')
        biases = tf.Variable(tf.constant(0.01, shape=[shape[3]]), name=name + 'biases')
        return weights, biases

    def create_fc_net(self, shape, name):
        weights = tf.Variable(tf.truncated_normal(shape=shape, stddev=0.01), name=name + 'weights')
        biases = tf.Variable(tf.constant(0.01, shape=[shape[1]]), name=name + 'biases')
        return weights, biases

    ## pooling層の場合
    # def weights_and_biases(self):
    #     return [self.conv1_weights, self.conv1_biases,
    #           self.fc1_weights, self.fc1_biases,
    #           self.fc2_weights, self.fc2_biases]

    ## 畳み込み層の場合
    def weights_and_biases(self):
        return [self.conv1_weights, self.conv1_biases,
            self.conv2_weights, self.conv2_biases,
            self.fc1_weights, self.fc1_biases,
            self.fc2_weights, self.fc2_biases]

    def copy_network_to(self, target, session):
        copy_operations = [target.assign(origin)
                          for origin, target in zip(self.weights_and_biases(), target.weights_and_biases())]
        session.run(copy_operations)

    def save_parameters(self, session, file_name, global_step):
        self.saver.save(session, save_path=file_name, global_step=global_step)

    def restore_parameters(self, session, file_name):
        self.saver.restore(session, save_path=file_name)
