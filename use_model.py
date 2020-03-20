
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

os.environ["TF_CPP_MIN_LOG_LEVEL"]='2'

class_dict = {0: "菊花", 1: "蒲公英", 2: "玫瑰", 3: "向日葵", 4: "郁金香" }

img_file = "./test_data/3.jpg"

def ckpt():
    with tf.Session() as sess:
        image_raw_data = tf.gfile.GFile(img_file, 'rb').read()
        image = tf.image.decode_jpeg(image_raw_data)
        if image.dtype != tf.float32:
            image = tf.image.convert_image_dtype(image, dtype=tf.float32)
        image = tf.image.resize_images(image, [224, 224])
        input_data = sess.run(image)
        x_input = [input_data]
        # 找到已有的模型，进行读取
        saver = tf.train.import_meta_graph('./m_model/m.ckpt.meta')
        saver.restore(sess, tf.train.latest_checkpoint('./m_model'))
        graph = tf.get_default_graph()
        x = graph.get_tensor_by_name("x-input:0")
        #batch = graph.get_tensor_by_name("batch-input:0")
        logits = graph.get_tensor_by_name("logits_eval:0")
        logits = tf.nn.softmax(logits)
        y = tf.argmax(logits, 1)
        softmax, predict = sess.run([logits, y], feed_dict={x: x_input})
        # 打印出预测矩阵
        print(softmax)
        print(predict, class_dict[predict[0]])
        img = plt.imread(img_file)
        plt.imshow(img)

def pb():
    with tf.Session() as sess:
        image_raw_data = tf.gfile.GFile(img_file, 'rb').read()
        image = tf.image.decode_jpeg(image_raw_data)
        if image.dtype != tf.float32:
            image = tf.image.convert_image_dtype(image, dtype=tf.float32)
        image = tf.image.resize_images(image, [224, 224])
        input_data = sess.run(image)
        x_input = [input_data]
        with tf.gfile.GFile('./m_model/m.pb', 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            sess.graph.as_default()
            tf.import_graph_def(graph_def, name='')  # 导入计算图
        # 需要有一个初始化的过程
        sess.run(tf.global_variables_initializer())
        # 输入
        x = sess.graph.get_tensor_by_name('x-input:0')

        softmax = sess.graph.get_tensor_by_name('softmax_eval:0')
        predict = sess.graph.get_tensor_by_name('predict_label:0')

        prob, label = sess.run([softmax, predict], feed_dict={x: x_input})
        print(prob)
        print(label, class_dict[label[0]])

pb()
