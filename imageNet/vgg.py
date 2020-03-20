
import tensorflow as tf

FC_SIZE = 256
NUM_CHANNEL = 3
NUM_LABEL = 5



def inference(input_tensor, is_training = False, regularizer=None, reuse=False):
    '''
    :param input_tensor:
    :param is_training:
    :param reuse:
    :param regularizer:
    :return:
    '''
    she = input_tensor.get_shape().as_list()
    print("input ", she)
    #1、卷积层
    with tf.variable_scope('conv1', reuse=reuse):
        with tf.variable_scope('conv1_1', reuse=reuse):
            conv1_1_weights = tf.get_variable("weight", [3, 3, NUM_CHANNEL, 64],
                initializer=tf.truncated_normal_initializer(stddev=0.1))
            conv1_1_biases = tf.get_variable("bias", [64], initializer=tf.constant_initializer(0.0))
            conv2d1_1 = tf.nn.conv2d(input_tensor, conv1_1_weights, strides=[1, 1, 1, 1], padding='SAME')
            conv1_1_out = tf.nn.relu(tf.nn.bias_add(conv2d1_1, conv1_1_biases))
        with tf.variable_scope('conv1_2', reuse=reuse):
            conv1_2_weights = tf.get_variable("weight", [3, 3, 64, 64],
                initializer=tf.truncated_normal_initializer(stddev=0.1))
            conv1_2_biases = tf.get_variable("bias", [64], initializer=tf.constant_initializer(0.0))
            conv2d1_2 = tf.nn.conv2d(conv1_1_out, conv1_2_weights, strides=[1, 1, 1, 1], padding='SAME')
            conv1_2_out = tf.nn.relu(tf.nn.bias_add(conv2d1_2, conv1_2_biases))
    #1 池化层
    with tf.name_scope("pool1"):
        pool1 = tf.nn.max_pool(conv1_2_out, ksize = [1,2,2,1],strides=[1,2,2,1],padding="SAME")

    # 2、卷积层
    with tf.variable_scope('conv2', reuse=reuse):
        with tf.variable_scope('conv2_1', reuse=reuse):
            conv2_1_weights = tf.get_variable("weight", [3, 3, 64, 128],
                initializer=tf.truncated_normal_initializer(stddev=0.1))
            conv2_1_biases = tf.get_variable("bias", [128], initializer=tf.constant_initializer(0.0))
            conv2d2_1 = tf.nn.conv2d(pool1, conv2_1_weights, strides=[1, 1, 1, 1], padding='SAME')
            conv2_1_out = tf.nn.relu(tf.nn.bias_add(conv2d2_1, conv2_1_biases))
        with tf.variable_scope('conv2_2', reuse=reuse):
            conv2_2_weights = tf.get_variable("weight", [3, 3, 128, 128],
                initializer=tf.truncated_normal_initializer(stddev=0.1))
            conv2_2_biases = tf.get_variable("bias", [128], initializer=tf.constant_initializer(0.0))
            conv2d2_2 = tf.nn.conv2d(conv2_1_out, conv2_2_weights, strides=[1, 1, 1, 1], padding='SAME')
            conv2_2_out = tf.nn.relu(tf.nn.bias_add(conv2d2_2, conv2_2_biases))
    #2 池化层
    with tf.name_scope("pool2"):
        pool2 = tf.nn.max_pool(conv2_2_out, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

    # 3、卷积层
    with tf.variable_scope('conv3', reuse=reuse):
        with tf.variable_scope('conv3_1', reuse=reuse):
            conv3_1_weights = tf.get_variable("weight", [3, 3, 128, 256],
                initializer=tf.truncated_normal_initializer(stddev=0.1))
            conv3_1_biases = tf.get_variable("bias", [256], initializer=tf.constant_initializer(0.0))
            conv2d3_1 = tf.nn.conv2d(pool2, conv3_1_weights, strides=[1, 1, 1, 1], padding='SAME')
            conv3_1_out = tf.nn.relu(tf.nn.bias_add(conv2d3_1, conv3_1_biases))
        with tf.variable_scope('conv3_2', reuse=reuse):
            conv3_2_weights = tf.get_variable("weight", [3, 3, 256, 256],
                initializer=tf.truncated_normal_initializer(stddev=0.1))
            conv3_2_biases = tf.get_variable("bias", [256], initializer=tf.constant_initializer(0.0))
            conv2d3_2 = tf.nn.conv2d(conv3_1_out, conv3_2_weights, strides=[1, 1, 1, 1], padding='SAME')
            conv3_2_out = tf.nn.relu(tf.nn.bias_add(conv2d3_2, conv3_2_biases))
        with tf.variable_scope('conv3_3', reuse=reuse):
            conv3_3_weights = tf.get_variable("weight", [3, 3, 256, 256],
                initializer=tf.truncated_normal_initializer(stddev=0.1))
            conv3_3_biases = tf.get_variable("bias", [256], initializer=tf.constant_initializer(0.0))
            conv2d3_3 = tf.nn.conv2d(conv3_2_out, conv3_3_weights, strides=[1, 1, 1, 1], padding='SAME')
            conv3_3_out = tf.nn.relu(tf.nn.bias_add(conv2d3_3, conv3_3_biases))
    #3 池化层
    with tf.name_scope("pool3"):
        pool3 = tf.nn.max_pool(conv3_3_out, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

    # 4、卷积层
    with tf.variable_scope('conv4', reuse=reuse):
        with tf.variable_scope('conv4_1', reuse=reuse):
            conv4_1_weights = tf.get_variable("weight", [3, 3, 256, 512],
                initializer=tf.truncated_normal_initializer(stddev=0.1))
            conv4_1_biases = tf.get_variable("bias", [512], initializer=tf.constant_initializer(0.0))
            conv2d4_1 = tf.nn.conv2d(pool3, conv4_1_weights, strides=[1, 1, 1, 1], padding='SAME')
            conv4_1_out = tf.nn.relu(tf.nn.bias_add(conv2d4_1, conv4_1_biases))
        with tf.variable_scope('conv4_2', reuse=reuse):
            conv4_2_weights = tf.get_variable("weight", [3, 3, 512, 512],
                initializer=tf.truncated_normal_initializer(stddev=0.1))
            conv4_2_biases = tf.get_variable("bias", [512], initializer=tf.constant_initializer(0.0))
            conv2d4_2 = tf.nn.conv2d(conv4_1_out, conv4_2_weights, strides=[1, 1, 1, 1], padding='SAME')
            conv4_2_out = tf.nn.relu(tf.nn.bias_add(conv2d4_2, conv4_2_biases))
        with tf.variable_scope('conv4_3', reuse=reuse):
            conv4_3_weights = tf.get_variable("weight", [3, 3, 512, 512],
                initializer=tf.truncated_normal_initializer(stddev=0.1))
            conv4_3_biases = tf.get_variable("bias", [512], initializer=tf.constant_initializer(0.0))
            conv2d4_3 = tf.nn.conv2d(conv4_2_out, conv4_3_weights, strides=[1, 1, 1, 1], padding='SAME')
            conv4_3_out = tf.nn.relu(tf.nn.bias_add(conv2d4_3, conv4_3_biases))
    #4 池化层
    with tf.name_scope("pool4"):
        pool4 = tf.nn.max_pool(conv4_3_out, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

    # 5、卷积层
    with tf.variable_scope('conv5', reuse=reuse):
        with tf.variable_scope('conv5_1', reuse=reuse):
            conv5_1_weights = tf.get_variable("weight", [3, 3, 512, 512],
                initializer=tf.truncated_normal_initializer(stddev=0.1))
            conv5_1_biases = tf.get_variable("bias", [512], initializer=tf.constant_initializer(0.0))
            conv2d5_1 = tf.nn.conv2d(pool4, conv5_1_weights, strides=[1, 1, 1, 1], padding='SAME')
            conv5_1_out = tf.nn.relu(tf.nn.bias_add(conv2d5_1, conv5_1_biases))
        with tf.variable_scope('conv5_2', reuse=reuse):
            conv5_2_weights = tf.get_variable("weight", [3, 3, 512, 512],
                initializer=tf.truncated_normal_initializer(stddev=0.1))
            conv5_2_biases = tf.get_variable("bias", [512], initializer=tf.constant_initializer(0.0))
            conv2d5_2 = tf.nn.conv2d(conv5_1_out, conv5_2_weights, strides=[1, 1, 1, 1], padding='SAME')
            conv5_2_out = tf.nn.relu(tf.nn.bias_add(conv2d5_2, conv5_2_biases))
        with tf.variable_scope('conv5_3', reuse=reuse):
            conv5_3_weights = tf.get_variable("weight", [3, 3, 512, 512],
                initializer=tf.truncated_normal_initializer(stddev=0.1))
            conv5_3_biases = tf.get_variable("bias", [512], initializer=tf.constant_initializer(0.0))
            conv2d5_3 = tf.nn.conv2d(conv5_2_out, conv5_3_weights, strides=[1, 1, 1, 1], padding='SAME')
            conv5_3_out = tf.nn.relu(tf.nn.bias_add(conv2d5_3, conv5_3_biases))
    #5 池化层、展平
    with tf.name_scope("pool5"):
        pool5 = tf.nn.max_pool(conv5_3_out, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
        pool5_shape = pool5.get_shape().as_list()
        nodes = pool5_shape[1] * pool5_shape[2] * pool5_shape[3]
        pool5_reshaped = tf.reshape(pool5, [-1, nodes])
        print("pool5_shape ", pool5_shape)

    #1、全连接层
    with tf.variable_scope('fc1', reuse=reuse):
        fc1_weights = tf.get_variable("weight", [nodes, FC_SIZE],
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))
        if regularizer != None:
            tf.add_to_collection('losses', regularizer(fc1_weights))
        fc1_biases = tf.get_variable("bias", [FC_SIZE], initializer=tf.constant_initializer(0.1))
        #fc1 = tf.nn.relu(tf.matmul(pool5_reshaped, fc1_weights) + fc1_biases)
        fc1 = tf.nn.relu_layer(pool5_reshaped,fc1_weights,fc1_biases)
        if is_training: fc1 = tf.nn.dropout(fc1, 0.5)
        sf = fc1.get_shape().as_list()
        print("fc1_shape ", sf)

    #2、全连接层
    with tf.variable_scope('fc2', reuse=reuse):
        fc2_weights = tf.get_variable("weight", [FC_SIZE, FC_SIZE],
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))
        if regularizer != None:
            tf.add_to_collection('losses', regularizer(fc2_weights))
        fc2_biases = tf.get_variable("bias", [FC_SIZE], initializer=tf.constant_initializer(0.1))
        #fc2 = tf.nn.relu(tf.nn.bias_add(tf.matmul(fc1, fc2_weights), fc2_biases))
        fc2 = tf.nn.relu_layer(fc1, fc2_weights, fc2_biases)
        if is_training: fc2 = tf.nn.dropout(fc2, 0.5)

    #3、输出层(全连接层)
    with tf.variable_scope('out', reuse=reuse):
        out_weights = tf.get_variable("weight", [FC_SIZE, NUM_LABEL],
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))
        if regularizer != None:
            tf.add_to_collection('losses', regularizer(out_weights))
        out_biases = tf.get_variable("bias", [NUM_LABEL], initializer=tf.constant_initializer(0.1))
        logits = tf.matmul(fc2, out_weights) + out_biases
        #softmax_logit = tf.nn.softmax(logit)
        shl = logits.get_shape().as_list()
        print("logit", shl)

    return logits


# %%
def conv(layer_name, x, out_channels, kernel_size=[3, 3], stride=[1, 1, 1, 1], is_pretrain=True):
    '''Convolution op wrapper, use RELU activation after convolution
    Args:
        layer_name: e.g. conv1, pool1...
        x: input tensor, [batch_size, height, width, channels]
        out_channels: number of output channels (or comvolutional kernels)
        kernel_size: the size of convolutional kernel, VGG paper used: [3,3]
        stride: A list of ints. 1-D of length 4. VGG paper used: [1, 1, 1, 1]
        is_pretrain: if load pretrained parameters, freeze all conv layers.
        Depending on different situations, you can just set part of conv layers to be freezed.
        the parameters of freezed layers will not change when training.
    Returns:
        4D tensor
    '''

    in_channels = x.get_shape()[-1]
    with tf.variable_scope(layer_name):
        w = tf.get_variable(name='weights',
                            trainable=is_pretrain,
                            shape=[kernel_size[0], kernel_size[1], in_channels, out_channels],
                            initializer=tf.contrib.layers.xavier_initializer())  # default is uniform distribution initialization
        b = tf.get_variable(name='biases',
                            trainable=is_pretrain,
                            shape=[out_channels],
                            initializer=tf.constant_initializer(0.0))
        x = tf.nn.conv2d(x, w, stride, padding='SAME', name='conv')
        x = tf.nn.bias_add(x, b, name='bias_add')
        x = tf.nn.relu(x, name='relu')
        return x


# %%
def pool(layer_name, x, kernel=[1, 2, 2, 1], stride=[1, 2, 2, 1], is_max_pool=True):
    '''Pooling op
    Args:
        x: input tensor
        kernel: pooling kernel, VGG paper used [1,2,2,1], the size of kernel is 2X2
        stride: stride size, VGG paper used [1,2,2,1]
        padding:
        is_max_pool: boolen
                    if True: use max pooling
                    else: use avg pooling
    '''
    if is_max_pool:
        x = tf.nn.max_pool(x, kernel, strides=stride, padding='SAME', name=layer_name)
    else:
        x = tf.nn.avg_pool(x, kernel, strides=stride, padding='SAME', name=layer_name)
    return x


# %%
def FC_layer(layer_name, x, out_nodes):
    '''Wrapper for fully connected layers with RELU activation as default
    Args:
        layer_name: e.g. 'FC1', 'FC2'
        x: input feature map
        out_nodes: number of neurons for current FC layer
    '''
    shape = x.get_shape()
    if len(shape) == 4:
        size = shape[1].value * shape[2].value * shape[3].value
    else:
        size = shape[-1].value

    with tf.variable_scope(layer_name):
        w = tf.get_variable('weights',
                            shape=[size, out_nodes],
                            initializer=tf.contrib.layers.xavier_initializer())
        b = tf.get_variable('biases',
                            shape=[out_nodes],
                            initializer=tf.constant_initializer(0.0))
        flat_x = tf.reshape(x, [-1, size])  # flatten into 1D

        x = tf.nn.bias_add(tf.matmul(flat_x, w), b)
        x = tf.nn.relu(x)
        return x


# %%
def vgg16(x, n_classes, is_pretrain=True):
    x = conv('conv1_1', x, 64, kernel_size=[3, 3], stride=[1, 1, 1, 1], is_pretrain=is_pretrain)
    x = conv('conv1_2', x, 64, kernel_size=[3, 3], stride=[1, 1, 1, 1], is_pretrain=is_pretrain)
    x = pool('pool1', x, kernel=[1, 2, 2, 1], stride=[1, 2, 2, 1], is_max_pool=True)

    x = conv('conv2_1', x, 128, kernel_size=[3, 3], stride=[1, 1, 1, 1], is_pretrain=is_pretrain)
    x = conv('conv2_2', x, 128, kernel_size=[3, 3], stride=[1, 1, 1, 1], is_pretrain=is_pretrain)
    x = pool('pool2', x, kernel=[1, 2, 2, 1], stride=[1, 2, 2, 1], is_max_pool=True)

    x = conv('conv3_1', x, 256, kernel_size=[3, 3], stride=[1, 1, 1, 1], is_pretrain=is_pretrain)
    x = conv('conv3_2', x, 256, kernel_size=[3, 3], stride=[1, 1, 1, 1], is_pretrain=is_pretrain)
    x = conv('conv3_3', x, 256, kernel_size=[3, 3], stride=[1, 1, 1, 1], is_pretrain=is_pretrain)
    x = pool('pool3', x, kernel=[1, 2, 2, 1], stride=[1, 2, 2, 1], is_max_pool=True)

    x = conv('conv4_1', x, 512, kernel_size=[3, 3], stride=[1, 1, 1, 1], is_pretrain=is_pretrain)
    x = conv('conv4_2', x, 512, kernel_size=[3, 3], stride=[1, 1, 1, 1], is_pretrain=is_pretrain)
    x = conv('conv4_3', x, 512, kernel_size=[3, 3], stride=[1, 1, 1, 1], is_pretrain=is_pretrain)
    x = pool('pool3', x, kernel=[1, 2, 2, 1], stride=[1, 2, 2, 1], is_max_pool=True)

    x = conv('conv5_1', x, 512, kernel_size=[3, 3], stride=[1, 1, 1, 1], is_pretrain=is_pretrain)
    x = conv('conv5_2', x, 512, kernel_size=[3, 3], stride=[1, 1, 1, 1], is_pretrain=is_pretrain)
    x = conv('conv5_3', x, 512, kernel_size=[3, 3], stride=[1, 1, 1, 1], is_pretrain=is_pretrain)
    x = pool('pool3', x, kernel=[1, 2, 2, 1], stride=[1, 2, 2, 1], is_max_pool=True)

    x = FC_layer('fc6', x, out_nodes=FC_SIZE)
    # x = tools.batch_norm(x)
    x = FC_layer('fc7', x, out_nodes=FC_SIZE)
    # x = tools.batch_norm(x)
    x = FC_layer('fc8', x, out_nodes=n_classes)

    return x