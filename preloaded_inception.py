import os
import time

import hickle as hkl
import numpy as np

from classify_image import extract_features_from_frame

os.makedirs("final_training_data", exist_ok=True)

for i in os.listdir(os.getcwd() + "/preinception_data/"):
    if i.endswith(".hkl") and "set" in i:
        number = i.split("-")[-1].replace(".hkl", "")
        print(number)
        train_set = hkl.load(os.getcwd() + "/preinception_data/" + i)
        print("Training Data:", train_set.shape)

        train_labels = hkl.load(os.getcwd() + "/final_training_data/" + "train_labels-" + str(number) + ".hkl")
        print("Train Labels:", train_labels.shape)

        data_set = None
        for example in range(len(train_set)):
            frame_representation = np.zeros((len(train_set[example]), 2048), dtype="float32")
            start = time.time()
            for frame in range(len(train_set[example])):
                frame_representation[frame] = extract_features_from_frame(train_set[example][frame])

            frame_representation = np.expand_dims(frame_representation, axis=0)
            print(" ###########  Time for clip ({} forward passes) ".format(len(train_set[example])), (time.time() - start))

            if example == 0:
                data_set = frame_representation
            else:
                data_set = np.concatenate((data_set, frame_representation), axis=0)

            print(data_set.shape)
        hkl.dump(data_set, "final_training_data/conv_features_train" + "-" + str(number) + ".hkl", mode="w", compression="gzip")
        print("Section Saved")
