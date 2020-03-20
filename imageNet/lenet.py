
import tensorflow as tf

FC_SIZE = 512
NUM_CHANNEL = 3
NUM_LABEL = 5

def inference(input_tensor, train, regularizer=None, reuse=False):
    """
    前向传播
    :param input_tensor: 四维：BATCH_SIZE,IMAGE_SIZE,IMAGE_SIZE,NUM_CHANNEL
    :param train: bool
    :param regularizer: 正则化函数
    :return: 二维矩阵：BATCH_SIZE，NUM_LABEL
    """
    she = input_tensor.get_shape().as_list()
    print("input ", she)
    #1、卷积层
    with tf.variable_scope('conv1', reuse=reuse):
        conv1_weights = tf.get_variable("weight", [3, 3, NUM_CHANNEL, 32],
            initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv1_biases = tf.get_variable("bias", [32], initializer=tf.constant_initializer(0.0))
        conv1 = tf.nn.conv2d(input_tensor, conv1_weights, strides=[1, 1, 1, 1], padding='SAME')
        relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_biases))
    #1、池化层
    with tf.name_scope("pool1"):
        pool1 = tf.nn.max_pool(relu1, ksize = [1,2,2,1],strides=[1,2,2,1],padding="SAME")

    #2、卷积层
    with tf.variable_scope("conv2", reuse=reuse):
        conv2_weights = tf.get_variable("weight", [3, 3, 32, 64],
            initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv2_biases = tf.get_variable("bias", [64], initializer=tf.constant_initializer(0.0))
        conv2 = tf.nn.conv2d(pool1, conv2_weights, strides=[1, 2, 2, 1], padding='SAME')
        relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_biases))
    #2、池化层
    with tf.name_scope("pool2"):
        pool2 = tf.nn.max_pool(relu2, ksize = [1,2,2,1],strides=[1,2,2,1],padding="SAME")

    #3、卷积层
    with tf.variable_scope("conv3", reuse=reuse):
        conv3_weights = tf.get_variable("weight", [3, 3, 64, 128],
            initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv3_biases = tf.get_variable("bias", [128], initializer=tf.constant_initializer(0.0))
        conv3 = tf.nn.conv2d(pool2, conv3_weights, strides=[1, 2, 2, 1], padding='SAME')
        relu3 = tf.nn.relu(tf.nn.bias_add(conv3, conv3_biases))
    #3、池化层
    with tf.name_scope("pool3"):
        pool3 = tf.nn.max_pool(relu3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        pool_shape = pool3.get_shape().as_list()
        nodes = pool_shape[1] * pool_shape[2] * pool_shape[3]
        reshaped = tf.reshape(pool3, [-1, nodes])

    #5、全连接层
    with tf.variable_scope('fc1', reuse=reuse):
        fc1_weights = tf.get_variable("weight", [nodes, FC_SIZE],
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))
        if regularizer is not None:
            tf.add_to_collection('losses', regularizer(fc1_weights))
        fc1_biases = tf.get_variable("bias", [FC_SIZE], initializer=tf.constant_initializer(0.1))
        fc1 = tf.nn.relu(tf.matmul(reshaped, fc1_weights) + fc1_biases)
        if train: fc1 = tf.nn.dropout(fc1, 0.5)
        sf = fc1.get_shape().as_list()
        print("fc1", sf)
    #6、输出层
    with tf.variable_scope('fc2', reuse=reuse):
        fc2_weights = tf.get_variable("weight", [FC_SIZE, NUM_LABEL],
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))
        if regularizer is not None:
            tf.add_to_collection('losses', regularizer(fc2_weights))
        fc2_biases = tf.get_variable("bias", [NUM_LABEL], initializer=tf.constant_initializer(0.1))
        logit = tf.matmul(fc1, fc2_weights) + fc2_biases
        shl = logit.get_shape().as_list()
        print("logit", shl)

    return logit

