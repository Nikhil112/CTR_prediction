import datetime
import time
import ConfigParser
import logging
import csv
import sys
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction import FeatureHasher
import numpy as np
import math
from numpy import genfromtxt
from StringIO import StringIO
from sklearn.preprocessing import OneHotEncoder
import tensorflow as tf	
from sklearn.metrics import roc_curve, auc, roc_auc_score
# import pdb

config = ConfigParser.ConfigParser()
config.read('CTR_model.cfg')

header_string = config.get('CTR_headers', 'file_headers')
# print header_string
header_dtype = config.get('CTR_headers', 'headers_dtype')
useful_feature = config.get('CTR_headers', 'useful_feature')
headers_CTR = header_string.split(',')
feature_dtype = header_dtype.split(',')
useful_features = useful_feature.split(',')
# np.set_printoptions(threshold='nan')


def feature_matrix(filename, headers_CTR, feature_dtype, useful_features):
	ctr_dict = []
	out_dict = []
	y_h = []
	header_size =  len(headers_CTR)
	with open(filename, 'rb') as csvfile:
		ctr_data = csv.reader(csvfile)
		for row in ctr_data:
			ctr_table = {}
			out_table = {}
			if len(row) == header_size and row[58] != 'None' and row[36] == 'cpc':
				for i in range(header_size):
					if useful_features[i] != '0':                                       
						if feature_dtype[i] == 'int':
							ctr_table[headers_CTR[i]] = int(row[i])
							out_table[headers_CTR[i]] = int(row[i])
						elif feature_dtype[i] == 'float':
							ctr_table[headers_CTR[i]] = float(row[i])
							out_table[headers_CTR[i]] = float(row[i])
						else: 	
							ctr_table[headers_CTR[i]+'_'+row[i]] = 1.0
							out_table[headers_CTR[i]] = row[i]
				ctr_dict.append(ctr_table)
				if row[59] == 'None':
					y_h.append(0.0)
					out_table[headers_CTR[59]] = 0.0
				else:
					y_h.append(1.0)
					out_table[headers_CTR[59]] = 1.0
				out_dict.append(out_table)

		
	# print len(ctr_dict)
	# print ctr_dict[0]
	hasher = FeatureHasher(n_features=2000, non_negative=True, input_type='dict')
	X_new = hasher.fit_transform(ctr_dict)
	x_h = X_new.toarray()
	y_h = np.asarray(y_h)
	print x_h.shape
	num_x = x_h.shape[0]
	y_h = y_h.reshape(num_x,1)
	print y_h.shape
	return x_h, y_h, out_dict


x_train, y_train, train_dict = feature_matrix('win_clk_data_2016062400.csv', headers_CTR, feature_dtype, useful_features)
x_test, y_test, test_dict = feature_matrix('csv_sample_2016060602.csv', headers_CTR, feature_dtype, useful_features)


def dict_csv(output_dict, filename):
    keys = output_dict[0].keys()
    with open(filename, 'wb') as output_file:
        dict_writer = csv.DictWriter(output_file, keys)
        dict_writer.writeheader()
        dict_writer.writerows(output_dict)

dict_csv(train_dict, 'train_feature.csv')
dict_csv(test_dict, 'test_feature.csv')

# Parameters
learning_rate = 0.001
training_epochs = 200
batch_size = 500
display_step = 20
num_example = x_train.shape[0]
# examples_to_show = 10

n_hidden = 2
n_input = 2000 
n_classes = 1

def data_next_batch(batch_size, index):
	start = index*batch_size
	end = start + batch_size
	return x_train[start:end], y_train[start:end]

# tf Graph input
x = tf.placeholder("float", [None, n_input])
y = tf.placeholder("float", [None, n_classes])

def change_dtype(in_array):
    out_array = in_array.astype(np.float64)
    return out_array
# Create model

weights1 = tf.Variable(tf.random_normal([n_input, n_hidden]))
weights2 = tf.Variable(tf.random_normal([n_hidden, n_classes]))
biases1 = tf.Variable(tf.random_normal([n_hidden]))
biases2 = tf.Variable(tf.random_normal([n_classes]))

# pred = tf.sigmoid(tf.matmul(tf.nn.sigmoid(tf.matmul(x, weights1) + biases1), weights2) + biases2)
pred = tf.add(tf.matmul((tf.matmul(x, weights1) + biases1), weights2), biases2)
out_pred = tf.sigmoid(pred)
# Define loss and optimizer
# cost = tf.reduce_mean(tf.nn.l2_loss((pred-y)))
cost = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(pred, y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, epsilon=0.01).minimize(cost)

# Initializing the variables
init = tf.initialize_all_variables()
# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    # Training cycle
    for epoch in range(training_epochs):
        # Loop over all batches
        avg_cost = 0.
        total_batch = int(num_example/batch_size)
        for i in range(total_batch):
            batch_xs, batch_ys = data_next_batch(batch_size, i)
            # Run optimization op (backprop) and cost op (to get loss value)
            # pdb.set_trace()
            sess.run(optimizer,feed_dict={x: batch_xs, y: batch_ys})
            c = cost.eval({x: batch_xs, y: batch_ys})
            # _, c = sess.run([optimizer, cost], feed_dict={x: batch_xs,
            #                                               y: batch_ys})
            avg_cost += c / total_batch
        # Display logs per epoch step

        if epoch % display_step == 0:
            print "Epoch:", '%04d' % (epoch+1), "cost=", \
                "{:.9f}".format(avg_cost)
    print("Optimization Finished!")

    y_pred_train = out_pred.eval({x: x_train, y: y_train})
    y_pred_test = out_pred.eval({x: x_test, y: y_test})
    np.savetxt("foo.csv", y_pred_train, delimiter=",")
    np.savetxt("foo1.csv", y_pred_test, delimiter=",")
    print "train_roc", roc_auc_score(y_train, y_pred_train)
    print "test_roc", roc_auc_score(y_test, y_pred_test)