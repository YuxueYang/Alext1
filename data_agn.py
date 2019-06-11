import tensorflow as tf
import tensorlayer as tl
import os
from PIL import Image
import os
import numpy as np
from tensorlayer.lazy_imports import LazyImport
import matplotlib
import matplotlib.pyplot as plt
cv2 = LazyImport("cv2")
from matplotlib import pylab


import numpy as np
# X_train, y_train, X_test, y_test = tl.files.load_cifar10_dataset(shape=(-1, 32, 32, 3), plotable=False)
image = tl.vis.read_image('data/cat/img1.jpg')
#tl.visualize.frame(image)
# h, w, c = image.shape
# print(h)
# print(w)
# print(c)

img_path = 'data/cat/img4.jpg'


# # print(label)
# ## Convert the bytes back to image as follow:
# # image = Image.frombytes('RGB', (32, 32), img_raw)
# # image = np.fromstring(img_raw, np.float32)
# # image = image.reshape([32, 32, 3])
# tl.visualize.frame(np.asarray(image, dtype=np.uint8), second=1, saveable=False, name='frame', fig_idx=1236)
#
# # image = cv2.imread(img_path)
# image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# image = np.asarray(image, dtype=np.float32)
# # image = img_to_array(image)
# image = image.astype(int)  # 0~255转换为int
# plt.imshow(image)
# pylab.imshow()



#
# # image = np.asarray(image, dtype=np.float32)
# # print(image)
# # # image = image/255
# # plt.imshow(image)
# # print(image.shape)
# #
# # X_train = np.asarray(X_train, dtype=np.float32)
# # plt.imshow(X_train[0])
# # # #print(X_train)
# # print(X_train.shape)
#
#
#
#
# import matplotlib.pyplot as plt
# import tensorflow as tf
# filename = 'img2.jpg'
# with tf.gfile.FastGFile(filename, 'rb') as f:
#     image_buffer = f.read()
# image = tf.image.decode_jpeg(image_buffer)
# #image = tf.image.resize_images(image,(224,224))
# with tf.Session() as sess:
#     img = sess.run(image)
#     plt.imshow(img)


# #coding: utf-8
# import matplotlib.pyplot as plt
# import tensorflow as tf
#
# image_raw = tf.gfile.FastGFile('img2.jpg','rb').read()
# img = tf.image.decode_jpeg(image_raw)  #Tensor
#
# with tf.Session() as sess:
#    img_ = img.eval()
#    print(img_.shape)
#
# plt.figure(1)
# plt.imshow(img_)
# plt.show()
#



#
#
# # classes = ['/data/cat', '/data/dog']
# # cwd = os.getcwd()
# # print(classes)
# # writer = tf.io.TFRecordWriter("train.tfrecords")
# # for index, name in enumerate(classes):
# #     class_path = cwd + name + "/"
# #     for img_name in os.listdir(class_path):
# #         img_path = class_path + img_name
# #         img = Image.open(img_path)
# #         img = img.resize((224, 224))
# #         ## Visualize the image as follow:
# #         # tl.visualize.frame(I=img, second=5, saveable=False, name='frame', fig_idx=12836)
# #         ## Converts a image to bytes
# #         img_raw = img.tobytes()
# #         ## Convert the bytes back to image as follow:
# #         # image = Image.frombytes('RGB', (224,224), img_raw)
# #         # tl.visualize.frame(I=image, second=1, saveable=False, name='frame', fig_idx=1236)
# #         ## Write the data into TF format
# #         # image     : Feature + BytesList
# #         # label     : Feature + Int64List or FloatList
# #         # sentence  : FeatureList + Int64List , see Google's im2txt example
# #         example = tf.train.Example(features=tf.train.Features(feature={ # SequenceExample for seuqnce example
# #             "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[index])),
# #             'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw])),
# #         }))
# #         writer.write(example.SerializeToString())  # Serialize To String
# # writer.close()
# # raw_dataset = tf.data.TFRecordDataset("train.tfrecords")
# # for serialized_example in raw_dataset:
# #     example = tf.train.Example()  # SequenceExample for seuqnce example
# #     example.ParseFromString(serialized_example.numpy())
# #     img_raw = example.features.feature['img_raw'].bytes_list.value
# #     label = example.features.feature['label'].int64_list.value
# #     ## converts a image from bytes
# #     image1 = Image.frombytes('RGB', (224, 224), img_raw[0])
# #     # tl.visualize.frame(np.asarray(image), second=0.5, saveable=False, name='frame', fig_idx=1283)
# #     print(label)
# #
# # def distort_img(x):
# #     x = tl.prepro.flip_axis(x, axis=1, is_random=True)
# #     x = tl.prepro.crop(x, wrg=28, hrg=28, is_random=True)
# #     return x
# #
# #
# # cat = '/data/cat'
# #
# # results = tl.prepro.threading_data(cat[0:2], distort_img)
# #
# # print(results.shape)
# #
# # tl.vis.save_images(cat[0:2], [1, 2], '_original.png')
# # tl.vis.save_images(results[0:2], [1, 2], '_distorted.png')
# #
# # tl.visualize.images2d(image, second=1, saveable=False, name='batch', dtype=None, fig_idx=2020121)
#


