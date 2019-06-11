# -*- coding: utf-8 -*-
import tensorflow as tf
import tensorlayer as tl
import numpy as np
import time


tf.logging.set_verbosity("DEBUG")
tl.logging.set_verbosity("DEBUG")

X_train, y_train, X_val, y_val, X_test, y_test = \
    tl.files.load_mnist_dataset(shape=(-1, 28, 28, 1))

sess = tf.InteractiveSession()
# batch_size = 128
batch_size = 20

x = tf.placeholder(tf.float32, shape=[batch_size, 28, 28, 1])
y_ = tf.placeholder(tf.int64, shape=[batch_size])

# AlexNet structure :5-conv layer; 3-fc layer
inputs = tl.layers.InputLayer(x, name='input')
conv1 = tl.layers.Conv2d(inputs, n_filter=96, filter_size=(11, 11), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv1')
conv1 = tl.layers.MaxPool2d(conv1, filter_size=(3, 3), strides=(2, 2), padding='SAME', name='con1')
conv2 = tl.layers.Conv2d(conv1, n_filter=256, filter_size=(5, 5), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv2')
conv2 = tl.layers.MaxPool2d(conv2, filter_size=(3, 3), strides=(2, 2), padding='SAME', name='conv2')
conv3 = tl.layers.Conv2d(conv2, n_filter=384, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv3')
conv4 = tl.layers.Conv2d(conv3, n_filter=384, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv4')
conv5 = tl.layers.Conv2d(conv4, n_filter=256, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv5')
conv5 = tl.layers.MaxPool2d(conv5, filter_size=(3, 3), strides=(2, 2), padding='SAME', name='conv5')
fat = tl.layers.FlattenLayer(conv5, name='fat')
fc6 = tl.layers.DenseLayer(fat, 4096, act=tf.nn.relu, name='fc6')
fc6 = tl.layers.DropoutLayer(fc6, keep=0.5, name='fc6')
fc7 = tl.layers.DenseLayer(fc6, 4096, act=tf.nn.relu, name='fc7')
# fc8 = tl.layers.DenseLayer(fc7, 1000, act=tf.nn.relu, name='fc8')
output = tl.layers.DenseLayer(fc7, 10, act=tf.identity, name='output')

# output.print_layers()


y = output.outputs

cost = tl.cost.cross_entropy(y, y_, 'cost')

correct_prediction = tf.equal(tf.argmax(y, 1), y_)
acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# n_epoch = 200
n_epoch = 10
learning_rate = 0.0001
print_freq = 2

train_params = output.all_params
train_op = tf.train.AdamOptimizer(learning_rate).minimize(cost, var_list=train_params)

tl.layers.initialize_global_variables(sess)
print(train_params)
# why can't use layer.print_params()?
# output.print_params()
# output.print_layers()

print('   learning_rate: %f' % learning_rate)
print('   batch_size: %d' % batch_size)
print('~~~~~~~~~~~training~~~~~~~~~~~')

for epoch in range(n_epoch):
    start_time = time.time()
    # tl.iterate.minibatches()输入特征及其对应的标签的两个Numpy数列依次同步的迭代函数
    for X_train_a, y_train_a in tl.iterate.minibatches(X_train, y_train, batch_size, shuffle=True):
        feed_dict = {x: X_train_a, y_: y_train_a}
        feed_dict.update(output.all_drop)
        sess.run(train_op, feed_dict=feed_dict)

    if epoch + 1 == 1 or (epoch + 1) % print_freq == 0:
        print("Epoch %d of %d took %fs" % (epoch + 1, n_epoch, time.time() - start_time))
        train_loss, train_acc, n_batch = 0, 0, 0
        for X_train_a, y_train_a in tl.iterate.minibatches(X_train, y_train, batch_size, shuffle=True):
            dp_dict = tl.utils.dict_to_one(output.all_drop)
            feed_dict = {x: X_train_a, y_: y_train_a}
            feed_dict.update(dp_dict)
            err, ac = sess.run([cost, acc], feed_dict=feed_dict)
            train_loss += err
            train_acc += ac
            n_batch += 1
        print("   train loss: %f" % (train_loss / n_batch))
        print("   train acc: %f" % (train_acc / n_batch))

        val_loss, val_acc, n_batch = 0, 0, 0
        for X_val_a, y_val_a in tl.iterate.minibatches(X_val, y_val, batch_size, shuffle=True):
            dp_dict = tl.utils.dict_to_one(output.all_drop)
            feed_dict = {x: X_val_a, y_: y_val_a}
            feed_dict.update(dp_dict)
            err, ac = sess.run([cost, acc], feed_dict=feed_dict)
            val_loss += err
            val_acc += ac
            n_batch += 1
        print("   val loss: %f" % (val_loss / n_batch))
#         print("   val acc: %f" % (val_acc / n_batch))



print('~~~~~~~~~~~~Evaluation~~~~~~~~~~~~~~~~~~')
test_loss, test_acc, n_batch = 0, 0, 0
for X_test_a, y_test_a in tl.iterate.minibatches(X_test, y_test, batch_size, shuffle=True):
    dp_dict = tl.utils.dict_to_one(output.all_drop)
    feed_dict = {x: X_test_a, y_: y_test_a}
    feed_dict.update(dp_dict)
    err, ac = sess.run([cost, acc], feed_dict=feed_dict)
    test_loss += err
    test_acc += ac
    n_batch += 1
print("   test loss: %f" % (test_loss / n_batch))
print("   test acc: %f" % (test_acc / n_batch))



# -----------------------------融合模型-------------------------------------------
# def acc(_logits, y_batch):
#     # return np.mean(np.equal(np.argmax(_logits, 1), y_batch))
#     return tf.reduce_mean(
#         tf.cast(tf.equal(tf.argmax(_logits, 1), tf.convert_to_tensor(y_batch, tf.int64)), tf.float32), name='accuracy'
#     )


# tl.utils.fit(
#     output, train_op=tf.train.AdamOptimizer(learning_rate=0.0001), cost=tl.cost.cross_entropy, X_train=X_train,
#     y_train=y_train, acc=acc, batch_size=256, n_epoch=20, X_val=X_val, y_val=y_val, eval_train=True,
#     tensorboard_dir='./tb_log'
# )

# # test
# tl.utils.test(output, acc, X_test, y_test, batch_size=None, cost=tl.cost.cross_entropy)
#
# # evaluation
# _logits = tl.utils.predict(output, X_test)
# y_pred = np.argmax(_logits, 1)
# tl.utils.evaluation(y_test, y_pred, n_classes=10)
#
# # save network weights
# output.save_weights('model.h5')






