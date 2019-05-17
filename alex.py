import numpy as np
import tensorflow as tf
import tensorlayer as tl
# from tensorlayer.layers.core import set_keep
import time

X_train, y_train, X_val, y_val, X_test, y_test = \
    tl.files.load_mnist_dataset(shape=(-1, 28, 28, 1))

sess = tf.InteractiveSession()
batch_size = 128

x = tf.placeholder(tf.float32, shape=[batch_size, 28, 28, 1])
y_ = tf.placeholder(tf.int64, shape=[batch_size])

network = tl.layers.InputLayer(x, name='input')
network = tl.layers.Conv2d(network, 32, (5, 5), (1, 1), act=tf.nn.relu, padding='SAME', name='cnn1')
network = tl.layers.MaxPool2d(network, (2, 2), (2, 2), padding='SAME', name='pool1')
network = tl.layers.Conv2d(network, 64, (5, 5), (1, 1), act=tf.nn.relu, padding='SAME', name='cnn2')
network = tl.layers.MaxPool2d(network, (2, 2), (2, 2), padding='SAME', name='pool2')

network = tl.layers.FlattenLayer(network, name='flatten')
network = tl.layers.DropoutLayer(network, keep=0.5, name='drop1')
network = tl.layers.DenseLayer(network, 256, act=tf.nn.relu, name='relu1')
network = tl.layers.DropoutLayer(network, keep=0.5, name='drop2')
network = tl.layers.DenseLayer(network, 10, act=tf.identity, name='output')

y = network.outputs

cost = tl.cost.cross_entropy(y, y_, 'cost')

correct_prediction = tf.equal(tf.argmax(y, 1), y_)
acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# n_epoch = 200
n_epoch = 10
learning_rate = 0.0001
print_freq = 2

train_params = network.all_params
train_op = tf.train.AdamOptimizer(learning_rate).minimize(cost, var_list=train_params)

tl.layers.initialize_global_variables(sess)
network.print_params()
network.print_layers()

print('   learning_rate: %f' % learning_rate)
print('   batch_size: %d' % batch_size)
print('~~~~~~~~~~~training~~~~~~~~~~~')

for epoch in range(n_epoch):
    start_time = time.time()
    for X_train_a, y_train_a in tl.iterate.minibatches(X_train, y_train, batch_size, shuffle=True):
        feed_dict = {x: X_train_a, y_: y_train_a}
        feed_dict.update(network.all_drop)
        sess.run(train_op, feed_dict=feed_dict)

    if epoch + 1 == 1 or (epoch + 1) % print_freq == 0:
        print("Epoch %d of %d took %fs" % (epoch + 1, n_epoch, time.time() - start_time))
        train_loss, train_acc, n_batch = 0, 0, 0
        for X_train_a, y_train_a in tl.iterate.minibatches(X_train, y_train, batch_size, shuffle=True):
            dp_dict = tl.utils.dict_to_one(network.all_drop)  # disable noise layers
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
            dp_dict = tl.utils.dict_to_one(network.all_drop)
            feed_dict = {x: X_val_a, y_: y_val_a}
            feed_dict.update(dp_dict)
            err, ac = sess.run([cost, acc], feed_dict=feed_dict)
            val_loss += err
            val_acc += ac
            n_batch += 1
        print("   val loss: %f" % (val_loss / n_batch))
        print("   val acc: %f" % (val_acc / n_batch))
        try:
            tl.vis.CNN2d(network.all_params[0].eval(), second=50, saveable=True, name='cnn1_' + str(epoch + 1),
                         fig_idx=2012)
        except:
            print("You should change vis.CNN(), if you want to save the feature images for different dataset")

print('~~~~~~~~~~~~Evaluation~~~~~~~~~~~~~~~~~~')
test_loss, test_acc, n_batch = 0, 0, 0
for X_test_a, y_test_a in tl.iterate.minibatches(X_test, y_test, batch_size, shuffle=True):
    dp_dict = tl.utils.dict_to_one(network.all_drop)
    feed_dict = {x: X_test_a, y_: y_test_a}
    feed_dict.update(dp_dict)
    err, ac = sess.run([cost, acc], feed_dict=feed_dict)
    test_loss += err
    test_acc += ac
    n_batch += 1
print("   test loss: %f" % (test_loss / n_batch))
print("   test acc: %f" % (test_acc / n_batch))
