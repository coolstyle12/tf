#-*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import pandas as pd
import math
from matplotlib import pyplot as plt
import time

t1 = time.time()

def read_data(file1, file2, file3, file4):
    normal = pd.read_csv(file1, delimiter="\t", names=["ch_1", "ch_2", "ch_3", "ch_4", "labels"])
    normal.labels = 1  # [1, 0, 0, 0]
    # rubbing
    rubbing = pd.read_csv(file2, delimiter="\t", names=["ch_1", "ch_2", "ch_3", "ch_4", "labels"])
    rubbing.labels = 2  # [0, 1, 0, 0]
    # misalignment
    misali = pd.read_csv(file3, delimiter="\t", names=["ch_1", "ch_2", "ch_3", "ch_4", "labels"])
    misali.labels = 3  # [0, 0, 1, 0]
    # oilwhirl
    oilwhirl = pd.read_csv(file4, delimiter="\t", names=["ch_1", "ch_2", "ch_3", "ch_4", "labels"])
    oilwhirl.labels = 4  # [0, 0, 0, 1]

    return normal, rubbing, misali, oilwhirl

def arrange(part1, part2, part3, part4):
    frames = [part1, part2, part3, part4]
    data_1 = pd.concat(frames)
    data_2 = data_1.iloc[np.random.permutation(len(data_1))]
    data_2['i'] = range(0, len(data_2))
    data_3 = data_2.set_index('i') # reset index

    labels = labeling(data_3, 'labels')
    data_4 = data_3.drop(['labels'], axis=1) # delete 'labels' columns
    data = data_4.as_matrix(columns=data_4.columns[0:])

    return data, labels

def labeling(data, column):
    tmp = []
    for i in range(len(data)):
        if data[column][i] == 1:
            tmp.append([1., 0, 0, 0]) # normal
        elif data[column][i] == 2:
            tmp.append([0, 1., 0, 0]) # rubbing
        elif data[column][i] == 3:
            tmp.append([0, 0, 1., 0]) # misalignment
        else:
            tmp.append([0, 0, 0, 1.]) # oilwhirl
    label = np.reshape(tmp, [-1, 4])
    return label

def split_data(data, val_size, test_size):
    """
    splits data to training, validation and testing parts
    """
    ntest = int(round(len(data) * (1 - test_size)))
    nval = int(round(len(data[:ntest]) * (1- val_size)))

    df_train, df_val, df_test = data[:nval], data[nval:ntest], data[ntest:]
    return df_train, df_val, df_test, nval, ntest,

def split_label(labels, nval, ntest):
    train_label,  val_label, test_label  = labels[:nval], labels[nval:ntest], labels[ntest:]
    return train_label, val_label, test_label

def xavier_init(n_inputs, n_outputs, uniform=True):
    """
    Set the parameter initializatino using the method described/
    This method is designed to keep the scale of the gradients roughly the same in all layers.
    Xavier Glorot and Yoshua Bengio (2010):
        Understanding the difficulty of training deep feedforward nerual networks. International
        conference on artificial intelligence and statistics.
    :param n_inputs:  The number of input nodes into each output.
    :param n_outputs: The number of output nodes for each input.
    :param uniform: If true use a uniform distribution, otherwise use a normal dist.
    :return: An initializer.
    """
    if uniform:
        # 6 was used in the paper
        init_range = math.sqrt(2.0/(n_inputs+n_outputs))
        return tf.random_uniform_initializer(-init_range, init_range)
    else:
        # 3 gives us approximately the same limits as aobve since this repicks
        # values greater than 2 standard deviations from the mean.
        stddev = math.sqrt(3.0/(n_inputs+n_outputs))
        return tf.truncated_normal_initializer(stddev=stddev)

class ROTOR:
    def __init__(self):
        path = '/home/coolstyle12/PycharmProjects/학부 논문/test_data/set1'
        normal, rubbing, misali, oilwhirl = read_data(path+'/normal.lvm', path+'/rubbing.lvm', path+'/misalignment.lvm', path+'/oilwhirl.lvm')
        data, labels = arrange(normal, rubbing, misali, oilwhirl)
        train_data, valid_data, test_data, nval, ntest = split_data(data, 0., 0.2)
        train_labels, valid_labels, test_labels = split_label(labels, nval, ntest)

        class train:
            def __init__(self):
                self.data = []
                self.labels = []
                self.batch_counter = 0
            def next_batch(self, num):
                if self.batch_counter + num >= len(self.labels):
                    batch_data = self.data[self.batch_counter:]
                    batch_labels = self.labels[self.batch_counter:]
                    left = num-len(batch_labels)
                    batch_data.extend(self.data[:left])
                    batch_labels.extend(self.labels[:left])
                    self.batch_counter = left
                else:
                    batch_data = self.data[self.batch_counter:self.batch_counter+num]
                    batch_labels = self.labels[self.batch_counter:self.batch_counter+num]
                    self.batch_counter += num
                return (batch_data, batch_labels)
        class valid:
            def __init__(self):
                self.data = []
                self.labels = []
        class test:
            def __init__(self):
                self.data = []
                self.labels = []

        self.train = train()
        self.valid = valid()
        self.test = test()
        self.train.data = train_data
        self.train.labels = train_labels
        self.valid.data = valid_data
        self.valid.labels = valid_labels
        self.test.data = test_data
        self.test.labels = test_labels

# Hyperparameter
learning_rate = 1e-4
batch_size = 50
reg_scale = 1e-4

# Input data
rotor = ROTOR()

# Set the model
X = tf.placeholder("float", [None, 4])
Y = tf.placeholder("float", [None, 4])
keep_prob = tf.placeholder("float", name="dropout")

W1 = tf.get_variable("W1", shape=[4, 512], initializer=xavier_init(4, 512))
W2 = tf.get_variable("W2", shape=[512, 512], initializer=xavier_init(512, 512))
W3 = tf.get_variable("W3", shape=[512, 512], initializer=xavier_init(512, 512))
W4 = tf.get_variable("W4", shape=[512, 512], initializer=xavier_init(512, 512))
W5 = tf.get_variable("W5", shape=[512, 512], initializer=xavier_init(512, 512))
W6 = tf.get_variable("W6", shape=[512, 512], initializer=xavier_init(512, 512))
W7 = tf.get_variable("W7", shape=[512, 512], initializer=xavier_init(512, 512))
W8 = tf.get_variable("W8", shape=[512, 512], initializer=xavier_init(512, 512))
W9 = tf.get_variable("W9", shape=[512, 512], initializer=xavier_init(512, 512))
W10 = tf.get_variable("W10", shape=[512, 512], initializer=xavier_init(512, 512))
W11 = tf.get_variable("W11", shape=[512, 4], initializer=xavier_init(512, 4))
b1 = tf.Variable(tf.constant(0.1, shape=[512]), name='bias1')
b2 = tf.Variable(tf.constant(0.1, shape=[512]), name='bias2')
b3 = tf.Variable(tf.constant(0.1, shape=[512]), name='bias3')
b4 = tf.Variable(tf.constant(0.1, shape=[512]), name='bias4')
b5 = tf.Variable(tf.constant(0.1, shape=[512]), name='bias5')
b6 = tf.Variable(tf.constant(0.1, shape=[512]), name='bias6')
b7 = tf.Variable(tf.constant(0.1, shape=[512]), name='bias7')
b8 = tf.Variable(tf.constant(0.1, shape=[512]), name='bias8')
b9 = tf.Variable(tf.constant(0.1, shape=[512]), name='bias9')
b10 = tf.Variable(tf.constant(0.1, shape=[512]), name='bias10')
b11 = tf.Variable(tf.constant(0.1, shape=[4]), name='bias11')

""" test accuracy: 0.6464
W1 = tf.get_variable("W1", shape=[4, 128], initializer=xavier_init(4, 128))
W2 = tf.get_variable("W2", shape=[128, 256], initializer=xavier_init(128, 256))
W3 = tf.get_variable("W3", shape=[256, 512], initializer=xavier_init(256, 512))
W4 = tf.get_variable("W4", shape=[512, 512], initializer=xavier_init(512, 512))
W5 = tf.get_variable("W5", shape=[512, 512], initializer=xavier_init(512, 512))
W6 = tf.get_variable("W6", shape=[512, 512], initializer=xavier_init(512, 512))
W7 = tf.get_variable("W7", shape=[512, 512], initializer=xavier_init(512, 512))
W8 = tf.get_variable("W8", shape=[512, 512], initializer=xavier_init(512, 512))
W9 = tf.get_variable("W9", shape=[512, 256], initializer=xavier_init(512, 256))
W10 = tf.get_variable("W10", shape=[256, 128], initializer=xavier_init(256, 128))
W11 = tf.get_variable("W11", shape=[128, 4], initializer=xavier_init(128, 4))
b1 = tf.Variable(tf.constant(0.1, shape=[128]), name='bias1')
b2 = tf.Variable(tf.constant(0.1, shape=[256]), name='bias2')
b3 = tf.Variable(tf.constant(0.1, shape=[512]), name='bias3')
b4 = tf.Variable(tf.constant(0.1, shape=[512]), name='bias4')
b5 = tf.Variable(tf.constant(0.1, shape=[512]), name='bias5')
b6 = tf.Variable(tf.constant(0.1, shape=[512]), name='bias6')
b7 = tf.Variable(tf.constant(0.1, shape=[512]), name='bias7')
b8 = tf.Variable(tf.constant(0.1, shape=[512]), name='bias8')
b9 = tf.Variable(tf.constant(0.1, shape=[256]), name='bias9')
b10 = tf.Variable(tf.constant(0.1, shape=[128]), name='bias10')
b11 = tf.Variable(tf.constant(0.1, shape=[4]), name='bias11')
"""

with tf.name_scope('HLayer1') as scope:
    L1 = tf.nn.relu(tf.matmul(X, W1) + b1)
    L1_drop = tf.nn.dropout(L1, keep_prob=keep_prob)

with tf.name_scope('HLayer2') as scope:
    L2 = tf.nn.relu(tf.matmul(L1_drop, W2) + b2)
    L2_drop = tf.nn.dropout(L2, keep_prob=keep_prob)

with tf.name_scope('HLayer3') as scope:
    L3= tf.nn.relu(tf.matmul(L2_drop, W3) + b3)
    L3_drop = tf.nn.dropout(L3, keep_prob=keep_prob)

with tf.name_scope('HLayer4') as scope:
    L4 = tf.nn.relu(tf.matmul(L3_drop, W4) + b4)
    L4_drop = tf.nn.dropout(L4, keep_prob=keep_prob)

with tf.name_scope('HLayer5') as scope:
    L5 = tf.nn.relu(tf.matmul(L4_drop, W5) + b5)
    L5_drop = tf.nn.dropout(L5, keep_prob=keep_prob)

with tf.name_scope('HLayer6') as scope:
    L6 = tf.nn.relu(tf.matmul(L5_drop, W6) + b6)
    L6_drop = tf.nn.dropout(L6, keep_prob=keep_prob)

with tf.name_scope('HLayer7') as scope:
    L7 = tf.nn.relu(tf.matmul(L6_drop, W7) + b7)
    L7_drop = tf.nn.dropout(L7, keep_prob=keep_prob)

with tf.name_scope('HLayer8') as scope:
    L8 = tf.nn.relu(tf.matmul(L7_drop, W8) + b8)
    L8_drop = tf.nn.dropout(L8, keep_prob=keep_prob)

with tf.name_scope('HLayer9') as scope:
    L9 = tf.nn.relu(tf.matmul(L8_drop, W9) + b9)
    L9_drop = tf.nn.dropout(L9, keep_prob=keep_prob)

with tf.name_scope('HLayer10') as scope:
    L10 = tf.nn.relu(tf.matmul(L9_drop, W10) + b10)
    L10_drop = tf.nn.dropout(L10, keep_prob=keep_prob)

with tf.name_scope("HLayer11") as scope:
    hypothesis = tf.nn.softmax(tf.matmul(L10_drop, W11) + b11)

# Add histogram
W1_hist = tf.histogram_summary("weight1", W1)
W2_hist = tf.histogram_summary("weight2", W2)
W3_hist = tf.histogram_summary("weight3", W3)
W4_hist = tf.histogram_summary("weight4", W4)
W5_hist = tf.histogram_summary("weight5", W5)
W6_hist = tf.histogram_summary("weight6", W6)
W7_hist = tf.histogram_summary("weight7", W7)
W8_hist = tf.histogram_summary("weight8", W8)
W9_hist = tf.histogram_summary("weight9", W9)
W10_hist = tf.histogram_summary("weight10", W10)
W11_hist = tf.histogram_summary("weight11", W11)
b1_hist = tf.histogram_summary("bias1", b1)
b2_hist = tf.histogram_summary("bias2", b2)
b3_hist = tf.histogram_summary("bias3", b3)
b4_hist = tf.histogram_summary("bias4", b4)
b5_hist = tf.histogram_summary("bias5", b5)
b6_hist = tf.histogram_summary("bias6", b6)
b7_hist = tf.histogram_summary("bias7", b7)
b8_hist = tf.histogram_summary("bias8", b8)
b9_hist = tf.histogram_summary("bias9", b9)
b10_hist = tf.histogram_summary("bias10", b10)
b11_hist = tf.histogram_summary("bias11", b11)

with tf.name_scope('cross_entropy') as scope:
    # W11_reshape = tf.reshape(W11, [4, 512])
    # W_total = tf.concat(0, [W1, W2, W3, W4, W5, W6, W7, W8, W9, W10, W11_reshape])
    W_total = tf.concat(0, [W2, W3, W4, W5, W6])
    L2_reg = reg_scale * tf.reduce_sum(tf.square(W_total))
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(hypothesis, Y)) + L2_reg
    ce_summ = tf.scalar_summary("cross_entropy", cross_entropy)

with tf.name_scope('train') as scope:
    train = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy) # learning_rate = 0.0001

with tf.name_scope('test') as scope:
    correct_prediction = tf.equal(tf.argmax(Y, 1), tf.argmax(hypothesis, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    accuracy_summ = tf.scalar_summary("Accuracy", accuracy)

# Another test data
path = '/home/coolstyle12/PycharmProjects/학부 논문/test_data/set2'
normal2, rubbing2, misali2, oilwhirl2 = read_data(path+'/normal.lvm', path+'/rubbing.lvm', path+'/misalignment.lvm', path+'/oilwhirl.lvm')
test_data2, test_labels2 = arrange(normal2, rubbing2, misali2, oilwhirl2)

path = '/home/coolstyle12/PycharmProjects/학부 논문/test_data/set3'
normal3, rubbing3, misali3, oilwhirl3 = read_data(path+'/normal.lvm', path+'/rubbing.lvm', path+'/misalignment.lvm', path+'/oilwhirl.lvm')
test_data3, test_labels3 = arrange(normal3, rubbing3, misali3, oilwhirl3)

path = '/home/coolstyle12/PycharmProjects/학부 논문/test_data/set4'
normal4, rubbing4, misali4, oilwhirl4 = read_data(path+'/normal.lvm', path+'/rubbing.lvm', path+'/misalignment.lvm', path+'/oilwhirl.lvm')
test_data4, test_labels4 = arrange(normal4, rubbing4, oilwhirl4, oilwhirl4)

path = '/home/coolstyle12/PycharmProjects/학부 논문/test_data/set5'
normal5, rubbing5, misali5, oilwhirl5 = read_data(path+'/normal.lvm', path+'/rubbing.lvm', path+'/misalignment.lvm', path+'/oilwhirl.lvm')
test_data5, test_labels5 = arrange(normal5, rubbing5, misali5, oilwhirl5)

# Start the graph
with tf.Session() as sess:
    tf.initialize_all_variables().run()
    merged = tf.merge_all_summaries()
    writer = tf.train.SummaryWriter("/tmp/DNN_logs", sess.graph)
    for step in range(5001):
        batch_xs, batch_ys = rotor.train.next_batch(batch_size)
        sess.run(train, feed_dict={X: batch_xs , Y: batch_ys, keep_prob: 0.5})
        if step%500 == 0:
            summary = sess.run(merged, feed_dict={X: batch_xs, Y: batch_ys, keep_prob: 1.0})
            writer.add_summary(summary, step)
            print ("step: %d, cost: %g, train_accuracy: %g" % (step, sess.run(cross_entropy, feed_dict={X: batch_xs, Y: batch_ys, keep_prob: 0.5}), sess.run(accuracy, feed_dict={X: batch_xs, Y: batch_ys, keep_prob: 1.0})))
    t2 = time.time()

    print ("test accuracy of set1: %g" % sess.run(accuracy, feed_dict={X: rotor.test.data, Y: rotor.test.labels, keep_prob: 1.0}))
    print ("test accuracy of set2: %g" % sess.run(accuracy, feed_dict={X: test_data2, Y: test_labels2, keep_prob: 1.0}))
    print ("test accuracy of set3: %g" % sess.run(accuracy, feed_dict={X: test_data3, Y: test_labels3, keep_prob: 1.0}))
    print ("test accuracy of set4: %g" % sess.run(accuracy, feed_dict={X: test_data4, Y: test_labels4, keep_prob: 1.0}))
    print ("test accuracy of set5: %g" % sess.run(accuracy, feed_dict={X: test_data5, Y: test_labels5, keep_prob: 1.0}))

print ("Process Finished! running time: %g" % (t2-t1))
#
# Plot
# plot_ch1_s = plt.plot(normal[:100], label='ch_1_side')
# plot_ch2_u = plt.plot(ch2_u.disp[:100], label='ch_2_upper')
# plot_ch3_s = plt.plot(ch3_s.disp[:100], label='ch_3_side')
# plot_ch4_u = plt.plot(ch4_u.disp[:100], label='ch_4_upper')
# legend = plt.legend(loc='upper right', shadow=True, fontsize=20)
# plt.show()
#
# #





















