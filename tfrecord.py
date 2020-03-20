
import os
import tensorflow as tf
import numpy as np
import time


os.environ["TF_CPP_MIN_LOG_LEVEL"]='2' # 只显示 warning 和 Error

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

# 将数据转化为tf.train.Example格式。
def _make_example(label, image):
    image = image.tobytes()
    example = tf.train.Example(features=tf.train.Features(feature={
        'label': _int64_feature(label),
        'image_raw': _bytes_feature(image)
    }))
    return example

#随机调整图片
def adjust_image(image_input, randint):
    if randint == 0:
        image = tf.image.flip_up_down(image_input) #上下翻转图片
        image = tf.image.random_brightness(image, max_delta=40. / 255.)#随机改变图片亮度
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)#饱和度
        image = tf.image.random_hue(image, max_delta=0.1)#色度
        image = tf.image.random_contrast(image, lower=0.8, upper=1.2)#对比度
    elif randint == 1:
        image = tf.image.flip_left_right(image_input)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_brightness(image, max_delta=40. / 255.)
        image = tf.image.random_contrast(image, lower=0.8, upper=1.2)
        image = tf.image.random_hue(image, max_delta=0.1)
    else:
        image = tf.image.transpose_image(image_input)
        image = tf.image.random_contrast(image, lower=0.8, upper=1.2)
        image = tf.image.random_brightness(image, max_delta=40. / 255.)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_hue(image, max_delta=0.1)
    return tf.clip_by_value(image, 0.0, 1.0)

def to_TFrecords(sess, datas_path, out_path):
    data_dirs = os.listdir(datas_path)
    label = 0
    for  class_dir in data_dirs:
        if class_dir.find(".") != -1:
            continue
        out_file_name = class_dir + ".tfrecords"
        out_file = os.path.join(out_path, out_file_name)
        img_files = os.listdir(os.path.join(datas_path, class_dir))
        print(out_file)
        with tf.python_io.TFRecordWriter(out_file) as writer:
            for img_file in img_files:
                file_name = os.path.join(datas_path, class_dir, img_file)
                try:
                    # 读取并解析图片，将图片转化为224*224
                    image_raw_data = tf.gfile.GFile(file_name, 'rb').read()
                    image = tf.image.decode_jpeg(image_raw_data)
                    if image.dtype != tf.float32:
                        image = tf.image.convert_image_dtype(image, dtype=tf.float32)
                    image = tf.image.resize_images(image, [224, 224])
                    image_value = sess.run(image)
                    example = _make_example(label, image_value)
                    writer.write(example.SerializeToString())
                except Exception as e:
                    print("{0}  error!!!".format(img_file), e)
                    #raise
                    continue
        label = label+1

#将图片reshape成224X224,并转化成float型，存储为tfrecord格式
def to_TFrecord_divide(sess, datas_path, out_path):
    data_dirs = os.listdir(datas_path)
    label = 0
    n_train = 0
    n_test = 0
    train_TFrecord = os.path.join(out_path, "train.tfrecords")
    test_TFrecord = os.path.join(out_path, "test.tfrecords")
    with tf.python_io.TFRecordWriter(train_TFrecord) as train_writer, \
            tf.python_io.TFRecordWriter(test_TFrecord) as test_writer:
        for  class_dir in data_dirs:
            if class_dir.find(".") != -1:
                continue
            img_files = os.listdir(os.path.join(datas_path, class_dir))
            print(label, class_dir, len(img_files))
            for img_file in img_files:
                file_name = os.path.join(datas_path, class_dir, img_file)
                #rand = np.random.randint(100)
                try:
                    image_raw_data = tf.gfile.GFile(file_name, 'rb').read()
                    image = tf.image.decode_jpeg(image_raw_data)
                    if image.dtype != tf.float32:
                        image = tf.image.convert_image_dtype(image, dtype=tf.float32)

                    #image_new = adjust_image(image, np.random.randint(3))

                    image = tf.image.resize_images(image, [224,224])
                    image = sess.run(image)
                    example = _make_example(label, image)
                    train_writer.write(example.SerializeToString())
                    n_train += 1
                    if n_train % 100 == 0: train_writer.flush()
                except Exception as e:
                    print("{0}  error!!!".format(img_file), e)
                    raise
                    continue
            label = label+1
    with open("./dataset/num_data.txt", 'w') as f:
        f.write(str(n_train) + " "+str(n_test))

with tf.Session() as sess:
    t = time.time()
    to_TFrecord_divide(sess, "./dataset/flower_photos/", "./dataset/tfrecords/")
    print("gross time is ", (time.time()-t)/60)
    
    
    
    
    