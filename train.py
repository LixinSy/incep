# -*- coding: utf-8 -*-

import os
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import graph_util
import time
#import matplotlib
#import matplotlib.pyplot as plt

from imageNet import vgg, lenet

os.environ["TF_CPP_MIN_LOG_LEVEL"]='3' # 只显示 warning 和 Error
#matplotlib.rcParams['font.family'] = "SimHei"
#matplotlib.rcParams['axes.unicode_minus'] = False


IMAGE_SIZE = 224
NUM_CHANNEL = 3
NUM_LABEL = 5

NUM_EPOCHS = 10
BATCH_SIZE = 32  #2的指数
LEARNING_RATE_BASE = 0.0003
LEARNING_RATE_DECAY = 0.99
REGULARIZATION_RATE = 0.0001
TRAIN_STEPS = 1000
MOVING_AVERAGE_DECAY = 0.99


def parser(record):
    features = tf.parse_single_example(
        record,
        features={
            'label':tf.FixedLenFeature([],tf.int64),
            'image_raw':tf.FixedLenFeature([],tf.string)
        })
    image = tf.decode_raw(features['image_raw'], tf.float32)
    image = tf.reshape(image, [IMAGE_SIZE, IMAGE_SIZE, 3])
    label = features['label']
    return image, label

def create_iterator(tfrecord_files, parser, batch_size, shuffle_buffer_size, num_epoch):
    dataset = tf.data.TFRecordDataset(tfrecord_files)
    dataset = dataset.map(parser)
    dataset = dataset.shuffle(shuffle_buffer_size).batch(batch_size)
    dataset = dataset.repeat(num_epoch)
    # 定义测试数据上的迭代器
    iterator = dataset.make_initializable_iterator()
    return iterator
    
def get_batch(iterator):
    image_batch, label_batch = iterator.get_next()
    return image_batch, label_batch

def train(train_files, test_files):
    # 定义输出为4维矩阵的placeholder
    x = tf.placeholder(tf.float32, [
        None,
        IMAGE_SIZE,
        IMAGE_SIZE,
        NUM_CHANNEL],
        name='x-input')
    y_ = tf.placeholder(tf.int64, [None], name='y-input')

    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
    y = lenet.inference(x, False, regularizer, False)
    #y = vgg.inference(x, False, regularizer, False)
    global_step = tf.Variable(0, trainable=False)

    # 定义损失函数、学习率、滑动平均操作以及训练过程。
    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())

    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=y_)
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))
    learning_rate = tf.train.exponential_decay(
        LEARNING_RATE_BASE,
        global_step,
        3300 * NUM_EPOCHS  / BATCH_SIZE, LEARNING_RATE_DECAY,
        staircase=True)
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
    #train_step = tf.train.MomentumOptimizer(learning_rate=learning_rate,momentum=0.9, use_nesterov=True)\
     #       .minimize(loss, global_step=global_step)
    with tf.control_dependencies([train_step, variables_averages_op]):
        train_op = tf.no_op(name='train')

    softmax_eval = tf.nn.softmax(y, name="softmax_eval")
    predict_label = tf.argmax(softmax_eval, 1, name="predict_label")
    # 计算正确率
    #validate_y = inference(x, False, None, True)
    correct_prediction = tf.equal(predict_label, y_)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # 外部使用模型的tensor
    #one = tf.constant(value=1, dtype=tf.float32)
    #logits_eval = tf.multiply(y, one, name='logits_eval')

    #收集训练过程的准确率
    x_times = []
    y_acc = []

    # 初始化TensorFlow持久化类。
    saver = tf.train.Saver()
    with tf.Session() as sess:
        # TFrecord dataset
        train_iterator = create_iterator(train_files, parser, BATCH_SIZE, 50000, NUM_EPOCHS)
        test_iterator = create_iterator(test_files, parser, BATCH_SIZE, 5000, NUM_EPOCHS)
        #
        tf.global_variables_initializer().run()
        sess.run(train_iterator.initializer)
        sess.run(test_iterator.initializer)
        train_imgs, train_labs = get_batch(train_iterator)
        test_imgs, test_labs = get_batch(test_iterator)
        # 循环进行训练，直到数据集完成输入、抛出OutOfRangeError错误
        for i in range(TRAIN_STEPS):
            try:
                xs, ys = sess.run([train_imgs, train_labs])
                lo, _ = sess.run([loss, train_op], feed_dict = {x:xs, y_:ys})

                if i % 100 == 0 or i+1 == TRAIN_STEPS:
                    xt, yt = sess.run([test_imgs, test_labs])
                    #print(yt, len(yt))
                    #if len(xt) < BATCH_SIZE: xt, yt = sess.run([test_imgs, test_labs])
                    acc = sess.run(accuracy, feed_dict = {x: xt, y_: yt})
                    print("after %d accuracy is %g, loss is %g" % (i, acc, lo))
                    x_times.append(i)
                    y_acc.append(acc)

            except Exception as e: # tf.errors.OutOfRangeError
                print(e)
                #raise
                #break
                pass
        try:
            #saver.save(sess, "./vgg_model/m.ckpt")
            constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph_def, [ 'softmax_eval', 'predict_label'])
            with tf.gfile.GFile("./m_model/m.pb", "wb") as f:
                f.write(constant_graph.SerializeToString())
        except Exception as e:
            print(e)
    '''
    try:#作图评估
        fig, ax = plt.subplots()
        ax.set_xlim([0, TRAIN_STEPS])
        ax.set_ylim([0., 1.0])
        plt.xlabel("X 训练次数")
        plt.ylabel("y 准确率")
        plt.plot(x_times, y_acc, color="red", label="验证准确率")
        plt.show()
    except Exception as e:
        print(e)
    '''
def main(argv=None):
    t = time.time()
    train_files = tf.train.match_filenames_once("./dataset/tfrecords/traind.tfrecords")
    test_files = tf.train.match_filenames_once("./dataset/tfrecords/testd.tfrecords")
    train("./dataset/tfrecords/traind.tfrecords", "./dataset/tfrecords/testd.tfrecords")
    print("gross time: ", (time.time()-t) / 60)

if __name__ == "__main__":
    try:
        main()
    except tf.errors.ResourceExhaustedError as e:
        print("资源耗尽 ", e)
        exit(-1)






