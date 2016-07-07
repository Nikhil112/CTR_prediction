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
import re
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


def hash_array(feature_dict, feature_num):
	# print feature_dict[0]
	if feature_num == 1:
		x_new = np.asarray(feature_dict)
		x_h = x_new.reshape(len(feature_dict), 1)
	else:
		hasher = FeatureHasher(n_features=feature_num, non_negative=True, input_type='dict')
		X_new = hasher.fit_transform(feature_dict)
		x_h = X_new.toarray()
		# vec = DictVectorizer()
		# x_h = vec.fit_transform(feature_dict).toarray()
		# print x_h.shape, type(x_h)
	return x_h

def feature_matrix(filename, headers_CTR, feature_dtype, i, feature_num):	
	y_h = []
	with open(filename, 'rb') as csvfile:
		ctr_data = csv.reader(csvfile)
		ctr_dict = []
		if i == 9:
			for row in ctr_data:
				if len(row) == 61 and row[58] != 'None' and row[36] == 'cpc':
					ctr_table = {}
					app_cat = re.split('-|#', row[i])
					ctr_table[headers_CTR[i]+'_'+app_cat[0]] = 1.0
					ctr_dict.append(ctr_table)
			x_h = hash_array(ctr_dict, feature_num)
		elif i == 26:
			for row in ctr_data:
				if len(row) == 61 and row[58] != 'None' and row[36] == 'cpc':
					ctr_table = {}
					try:
						day = datetime.datetime.strptime(row[i], '%Y-%m-%d').strftime('%A')
						ctr_table[headers_CTR[i]+'_'+day] = 1.0
						ctr_dict.append(ctr_table)
					except:
						print 'not a valid date'
			x_h = hash_array(ctr_dict, feature_num)
		else:
			for row in ctr_data:
				if len(row) == 61 and row[58] != 'None' and row[36] == 'cpc':
					if feature_dtype[i] == 'float':
						ctr_dict.append(float(row[i]))
					else:
						ctr_table = {}
						ctr_table[headers_CTR[i]+'_'+row[i]] = 1.0
						ctr_dict.append(ctr_table)
			x_h = hash_array(ctr_dict, feature_num)
				
	return x_h

def y_labels(filename, num_x):
	y_h = []
	with open(filename, 'rb') as csvfile:
		ctr_data = csv.reader(csvfile)
		for row in ctr_data:
			if len(row) == 61 and row[58] != 'None' and row[36] == 'cpc':
				if row[59] == 'None':
					y_h.append(0.0)
				else:
					y_h.append(1.0)
	y_h = np.asarray(y_h)
	y_h = y_h.reshape(num_x,1)
	return y_h

def combine_data(filename, headers_CTR, feature_dtype):		
	country = feature_matrix(filename, headers_CTR, feature_dtype, 1, 15)
	exchange = feature_matrix(filename, headers_CTR, feature_dtype, 2, 10)
	os_id = feature_matrix(filename, headers_CTR, feature_dtype, 3, 5)
	os_version = feature_matrix(filename, headers_CTR, feature_dtype, 4, 100)
	traffic = feature_matrix(filename, headers_CTR, feature_dtype, 5, 2)
	publisher = feature_matrix(filename, headers_CTR, feature_dtype, 6, 300)
	bundle_id = feature_matrix(filename, headers_CTR, feature_dtype, 7, 800)
	app_cat = feature_matrix(filename, headers_CTR, feature_dtype, 9, 20)
	con_type = feature_matrix(filename, headers_CTR, feature_dtype, 13, 3)
	device_type = feature_matrix(filename, headers_CTR, feature_dtype, 14, 5)
	position = feature_matrix(filename, headers_CTR, feature_dtype, 18, 10)
	date = feature_matrix(filename, headers_CTR, feature_dtype, 26, 7)
	device_js = feature_matrix(filename, headers_CTR, feature_dtype, 29, 2)
	dsp_id = feature_matrix(filename, headers_CTR, feature_dtype, 32, 7)
	adv_id = feature_matrix(filename, headers_CTR, feature_dtype, 33, 25)
	campaign_id = feature_matrix(filename, headers_CTR, feature_dtype, 34, 50)
	# creative_id = feature_matrix(filename, headers_CTR, feature_dtype, 35, 200)
	bid_floor = feature_matrix(filename, headers_CTR, feature_dtype, 39, 1)
	# feature_matrix('CTR_test.csv', headers_CTR, feature_dtype)

	x_train = np.concatenate((country, exchange, os_id, os_version, traffic, publisher, bundle_id, app_cat, 
				con_type, device_type, position, date, device_js, dsp_id, adv_id, campaign_id, bid_floor), axis=1)

	return x_train

x_train = combine_data('/home/nikhil/train_day.csv', headers_CTR, feature_dtype)
y_train = y_labels('/home/nikhil/train_day.csv', x_train.shape[0])
x_test = combine_data('/home/nikhil/test_day.csv', headers_CTR, feature_dtype)
y_test = y_labels('/home/nikhil/test_day.csv', x_test.shape[0])
print x_train.shape, y_train.shape, x_test.shape, y_test.shape
# Parameters
learning_rate = 0.001
training_epochs = 200
batch_size = 500
display_step = 20
num_example = x_train.shape[0]

# different layer nodes
n_hidden = 10 #1 hidden layer with 10 nodes
n_input = x_train.shape[1] 
n_classes = 1 #output layer with one node 

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


pred = tf.add(tf.matmul(tf.nn.relu(tf.matmul(x, weights1) + biases1), weights2), biases2)
out_pred = tf.sigmoid(pred)

# Define loss and optimizer
cost = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(pred, y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

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
            sess.run(optimizer,feed_dict={x: batch_xs, y: batch_ys})
            c = cost.eval({x: batch_xs, y: batch_ys})
            avg_cost += c / total_batch
        # Display logs per epoch step

        if epoch % display_step == 0:
            print "Epoch:", '%04d' % (epoch+1), "cost=", \
                "{:.9f}".format(avg_cost)
    print("Optimization Finished!")

    # computing train and test predicted value
    y_pred_train = out_pred.eval({x: x_train, y: y_train})
    y_pred_test = out_pred.eval({x: x_test, y: y_test})
    np.savetxt("/home/nikhil/foo.csv", y_pred_train, delimiter=",")
    np.savetxt("/home/nikhil/foo1.csv", y_pred_test, delimiter=",")
    print "train_roc", roc_auc_score(y_train, y_pred_train)
    print "test_roc", roc_auc_score(y_test, y_pred_test)