import tensorflow as tf
import numpy as np
import argparse
import time
import subprocess as sp
import os
import hickle as hkl
print tf.__version__

num_layers = 4
drop_out_prob = 0.8
batch_size = 30
epochs = 30
learning_rate = 0.00001
test_size = 800
validation_size = 600

videos_loaded = 0
for i in os.listdir(os.getcwd()):
    if i.endswith(".hkl"):
        if 'features' in i:
            print i
            if videos_loaded == 0:
                loaded = hkl.load(i)
            else:
                loaded = np.concatenate((loaded,hkl.load(i)), axis = 0)
            videos_loaded = videos_loaded + 1
            print loaded.shape

videos_loaded = 0
for i in os.listdir(os.getcwd()):
    if i.endswith(".hkl"):
        if "labels" in i:
            print i
            if videos_loaded == 0:
                labels = hkl.load(i)
            else:
                labels = np.concatenate((labels,hkl.load(i)), axis = 0)
            videos_loaded = videos_loaded + 1
            print labels.shape

loaded, labels = unison_shuffled_copies(loaded,labels)
print loaded.shape, labels.shape

test_set = loaded[:test_size]
test_labels = labels[:test_size]
validation_set = loaded[test_size:(validation_size+test_size)]
validation_labels = labels[test_size:(validation_size+test_size)]
test_set_size = len(test_set)
loaded = loaded[(test_size+validation_size):]
labels = labels[(test_size+validation_size):]
print "Test Set Shape: ", test_set.shape
print "Validation Set Shape: ", validation_set.shape
print "Training Set Shape: ", loaded.shape

hkl.dump(test_set, 'test_data.hkl', mode='w', compression='gzip', compression_opts=9)
hkl.dump(test_labels, 'test_lbls.hkl', mode='w', compression='gzip', compression_opts=9)

device_name = "/gpu:0"

with tf.device(device_name):
    tf.reset_default_graph()
    logs_path = '/tmp/4_d-0.8'
    display_step = 40
    n_input = 2048
    n_hidden = 1024
    n_classes = 3
    x = tf.placeholder("float32", [None, None, n_input])
    y = tf.placeholder("float32", [None, n_classes])
    input_batch_size = tf.placeholder("int32", None)
    weights = {
        'out': tf.Variable(tf.random_normal([n_hidden, n_classes]))
    }
    biases = {
        'out': tf.Variable(tf.random_normal([n_classes]))
    }

    def RNN(x, weights, biases):
        cell = tf.contrib.rnn.LSTMCell(n_hidden, state_is_tuple=True)
        cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=drop_out_prob)
        cell = tf.contrib.rnn.MultiRNNCell([cell] * num_layers, state_is_tuple=True)
        init_state = cell.zero_state(input_batch_size, tf.float32)
        outputs, states = tf.nn.dynamic_rnn(cell,x, initial_state = init_state, swap_memory = True)
        print states
        return tf.matmul(outputs[:,-1,:], weights['out']) + biases['out']

    with tf.name_scope('Model'):
        pred = RNN(x, weights, biases)
    print "prediction", pred
    with tf.name_scope('Loss'):
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
    with tf.name_scope('SGD'):
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
    with tf.name_scope('Accuracy'):
        correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    init = tf.global_variables_initializer()
    tf.summary.scalar("loss", cost)
    tf.summary.scalar("training_accuracy", accuracy)
    merged_summary_op = tf.summary.merge_all()

current_epochs = 0
with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
    saver = tf.train.Saver()
    sess.run(init)
    summary_writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())
    step = 0
    
    while step < (len(labels)/batch_size):
        batch_x = loaded[step*batch_size:(step+1)*batch_size]
        batch_y = labels[step*batch_size:(step+1)*batch_size]
        _,acc,loss,summary = sess.run([optimizer,accuracy,cost,merged_summary_op], feed_dict={x: batch_x, y: batch_y, input_batch_size:batch_size})
        print 'ran'
        summary_writer.add_summary(summary, step*batch_size+current_epochs * (len(labels)/batch_size)*batch_size)
        if step % display_step == 0:
            accuracies = []
            train_drop_out_prob = drop_out_prob
            drop_out_prob = 1.0
            for i in range(0,validation_size/batch_size):
                validation_batch_data = validation_set[i*batch_size:(i+1)*batch_size]
                validation_batch_labels = validation_labels[i*batch_size:(i+1)*batch_size]
                validation_batch_acc,_ = sess.run([accuracy,cost], feed_dict={x: validation_batch_data, y: validation_batch_labels, input_batch_size: batch_size})    
                accuracies.append(validation_batch_acc)
            summary = tf.Summary()
            summary.value.add(tag="Validation_Accuracy", simple_value=sum(accuracies)/len(accuracies))
            summary_writer.add_summary(summary, step*batch_size +current_epochs * (len(labels)/batch_size)*batch_size)
            saver.save(sess, 'fencing_AI_checkpoint')
            print "Validation Accuracy - All Batches:", sum(accuracies)/len(accuracies)
            drop_out_prob = train_drop_out_prob
            print "Iter " + str(step*batch_size+current_epochs * (len(labels)/batch_size)*batch_size) + ", Minibatch Loss= " + \
                  "{:.6f}".format(loss) + ", Train Accuracy= " + \
                  "{:.5f}".format(acc)
        step += 1
        if current_epochs < epochs:
            if step >= (len(labels)/batch_size):
                print "###################### New epoch ##########"
                current_epochs = current_epochs + 1
                learning_rate = learning_rate- (learning_rate*0.15)
                step = 0
                loaded, labels = unison_shuffled_copies(loaded,labels)
    print "Learning finished!"
