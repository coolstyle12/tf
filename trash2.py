#-*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import pandas as pd
import math
from matplotlib import pyplot as plt
import time

# t1 = time.time()

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

# def xavier_init(n_inputs, n_outputs, uniform=True):
#     """
#     Set the parameter initializatino using the method described/
#     This method is designed to keep the scale of the gradients roughly the same in all layers.
#     Xavier Glorot and Yoshua Bengio (2010):
#         Understanding the difficulty of training deep feedforward nerual networks. International
#         conference on artificial intelligence and statistics.
#     :param n_inputs:  The number of input nodes into each output.
#     :param n_outputs: The number of output nodes for each input.
#     :param uniform: If true use a uniform distribution, otherwise use a normal dist.
#     :return: An initializer.
#     """
#     if uniform:
#         # 6 was used in the paper
#         init_range = math.sqrt(2.0/(n_inputs+n_outputs))
#         return tf.random_uniform_initializer(-init_range, init_range)
#     else:
#         # 3 gives us approximately the same limits as aobve since this repicks
#         # values greater than 2 standard deviations from the mean.
#         stddev = math.sqrt(3.0/(n_inputs+n_outputs))
#         return tf.truncated_normal_initializer(stddev=stddev)

def weight_variable(shape, name):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial, name=name)

def bias_variable(shape, name):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name=name)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

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
batch_size = 100

# Input data
rotor = ROTOR()

# Set the model
X = tf.placeholder("float", [None, 4], name="x-input")
Y = tf.placeholder("float", [None, 4], name="y-input")
keep_prob = tf.placeholder("float", name="dropout")
x_reshape = tf.reshape(X, [-1, 20, 20, 1]) # 20 20 1

# W_conv1 = weight_variable([5, 5, 1, 32], name="weight1")
# W_conv2 = weight_variable([5, 5, 32, 64], name="weight2")
# W_fc1 = weight_variable([1600, 1024], name="weight3")
# W_fc2 = weight_variable([1024, 4], name="weight4")
# b_conv1 = bias_variable([32], name="bias1")
# b_conv2 = bias_variable([64], name="bias2")
# b_fc1= bias_variable([1024], name="bias3")
# b_fc2 = bias_variable([4], name="bias4")
#
# with tf.name_scope("Convlayer1") as scope:
#     h_conv1 = tf.nn.relu(conv2d(x_reshape, W_conv1) + b_conv1) # 20 20 32
#
# with tf.name_scope("2x2maxpooling") as scope:
#     h_pool1 = max_pool_2x2(h_conv1) # 10 10 32
#
# with tf.name_scope("COnvlayer2") as scope:
#     h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2) # 10 10 64
#
# with tf.name_scope("2x2maxpooling") as scope:
#     h_pool2 = max_pool_2x2(h_conv2) # 5 5 64
#
# with tf.name_scope("FClayer1") as scope:
#     h_pool2_flat = tf.reshape(h_pool2, [-1, 1600]) # -1*(5*5*64)
#     h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1) # -1*1024
#     h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
#
# with tf.name_scope("FClayer2") as scope:
#     y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
#
# with tf.name_scope("cross_entropy") as scope:
#     cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_conv, Y))
#     ce_summ = tf.scalar_summary("cross_entropy", cross_entropy)
#
# with tf.name_scope("train") as scope:
#     train = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)
#
# with tf.name_scope("accuracy") as scope:
#     correct_prediction = tf.equal(tf.argmax(Y, 1), tf.argmax(y_conv, 1))
#     accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
#     accuracy_summ = tf.scalar_summary("accuracy", accuracy)
#
# # Add histogram
# W_conv1_hist = tf.histogram_summary("wegiht1", W_conv1)
# W_conv2_hist = tf.histogram_summary("weight2", W_conv2)
# W_fc1_hist = tf.histogram_summary("weigth3", W_fc1)
# W_fc2_hist = tf.histogram_summary("weight4", W_fc2)
# b_conv1_hist = tf.histogram_summary("bias1", b_conv1)
# b_conv2_hist = tf.histogram_summary("bias2", b_conv2)
# b_fc1_hist = tf.histogram_summary("bias3", b_fc1)
# b_fc2_hist = tf.histogram_summary("bias4", b_fc2)

with tf.Session() as sess:
    batch_xs, batch_ys = rotor.train.next_batch(batch_size)
    a = tf.reshape(batch_xs, [-1, 20, 20, 1])
    print np.shape(batch_xs), np.shape(batch_ys), a.get_shape()
