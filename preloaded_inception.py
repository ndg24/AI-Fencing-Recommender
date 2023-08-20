import tensorflow as tf
import numpy as np
import os
from classify_image import *
import hickle as hkl
import time

FLAGS.model_dir = 'model/'
maybe_download_and_extract()
create_graph()

with tf.Session() as sess: 
    representation_tensor = sess.graph.get_tensor_by_name('pool_3:0')

    for i in os.listdir(os.getcwd() + "/preinception_data/"):
        if i.endswith(".hkl"):
            if "set" in i:
                number = i.split("-")[-1].replace(".hkl","")
                print number
                train_set = hkl.load(os.getcwd() + '/preinception_data/'+i)
                print "Training Data:", train_set.shape

                train_labels = hkl.load(os.getcwd() + '/final_training_data/'+"train_labels-" + str(number) + ".hkl")
                print "Train Labels:", train_labels.shape
                for example in range(len(train_set)):
                    frame_representation = np.zeros((len(train_set[example]), 2048), dtype='float32')
                    start = time.time()
                    for frame in range(len(train_set[example])):
                        rep = sess.run(representation_tensor, {'DecodeJpeg:0': train_set[example][frame]})
                        frame_representation[frame] = np.squeeze(rep)
                    
                    frame_representation = np.expand_dims(frame_representation,axis=0)    
                    print " ###########  Time for clip (21 forward passes) {} ", (time.time() - start)
                    
                    if example == 0:
                        data_set = frame_representation
                    else:
                        data_set = np.concatenate((data_set,frame_representation), axis = 0)
                   
                    print data_set.shape
                hkl.dump(data_set, 'final_training_data/conv_features_train' + '-' + str(number) +'.hkl', mode='w', compression='gzip', compression_opts=9)
                print "Section Saved"
