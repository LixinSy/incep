import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
import pandas as pd

os.environ["TF_CPP_MIN_LOG_LEVEL"]='3'

def get_datas(tfrecord_file):
    labels = []
    images = []
    with tf.Session() as sess:
        for example in tf.python_io.tf_record_iterator(tfrecord_file):
            features = tf.parse_single_example(example,
                features={
                    'label':tf.FixedLenFeature([],tf.float32),
                    'image_raw':tf.FixedLenFeature([],tf.string)
                })
            image = tf.decode_raw(features['image_raw'], tf.float32)
            image = tf.reshape(image, [299, 299, 3])
            image = tf.image.convert_image_dtype(image, dtype=tf.uint8)
            label = features['label']
            #tf.initialize_all_variables().run()
            im, l=sess.run([image, label])
            labels.append(l)
            images.append(im)
    d = pd.DataFrame()
    d['label'] = labels
    d['image'] = images
    return d

def parser(record):
    features = tf.parse_single_example(
        record,
        features={
            'label':tf.FixedLenFeature([],tf.float32),
            'image_raw':tf.FixedLenFeature([],tf.string)
        })
    image = tf.decode_raw(features['image_raw'], tf.float32)
    image = tf.reshape(image, [299, 299, 3])
    image = tf.image.convert_image_dtype(image, dtype=tf.uint8)
    labels = tf.cast(features['label'], tf.int32)
    
    return image, labels

def create_iterator(tfrecord_file):
    dataset = tf.data.TFRecordDataset(tfrecord_file)
    dataset = dataset.map(parser)
    dataset = dataset.shuffle(1000).batch(9)
    #dataset = dataset.shuffle(1000)
    dataset = dataset.repeat(5)
    # 定义测试数据上的迭代器
    iterator = dataset.make_initializable_iterator()
    return iterator
    
def get_b(iterator):
    
    image_batch, label_batch = iterator.get_next()
    return image_batch, label_batch


with tf.Session() as sess:
    '''
    iterator = create_iterator("./dataset/tfrecords/test.tfrecords")
    sess.run(iterator.initializer)
    for i in range(10):
        x, y = get_b(iterator)
        xs, ys = sess.run([x,y])
        print(ys)
    '''
    m = tf.one_hot([1,2,3,4,0,1], 5)
    arr = sess.run(m)
    print(arr)
    l = sess.run(tf.argmax(m, 1))
    print(l)
    
import matplotlib

matplotlib.rcParams['font.family'] = "SimHei"
matplotlib.rcParams['axes.unicode_minus'] = False

x_times=[2,4,6,8,10]
y_acc = [0.1,0.2,0.4,0.6,0.8]
fig, ax = plt.subplots()
ax.set_xlim([0, 10])
ax.set_ylim([0., 1.0])
plt.xlabel("X 训练次数")
plt.ylabel("y 准确率")
plt.plot(x_times, y_acc, color="red", label="验证准确率")
plt.show()



