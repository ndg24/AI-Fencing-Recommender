import cv2
import tensorflow as tf
import numpy as np
import argparse
import time
import cv
import subprocess as sp
import os
from pylab import *
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

green_box = cv2.imread("greenbox.png")
red_box = cv2.imread("redbox.png")
white_box = cv2.imread("whitebox.png")

import cPickle
with open('logistic_classifier_0-15.pkl', 'rb') as fid:
    model = cPickle.load(fid)

def check_lights(frame):
    leftOff = False
    leftOn = False
    rightOff = False
    rightOn = False
    string = ""
    if (np.sum(abs(frame[330:334, 140:260].astype(int) - red_box.astype(int))) <= 40000):
        string = string + "On"
    elif (np.sum(abs(frame[337:348, 234:250].astype(int) - white_box.astype(int))) <= 7000):
        string = string + "Off"
    else:
        string = string + "No"
    string = string + "-"
    if (np.sum(abs(frame[330:334, 380:500].astype(int) - green_box.astype(int))) <= 40000):
        string = string + "On"
    elif (np.sum(abs(frame[337:348, 390:406].astype(int) - white_box.astype(int))) <= 7000):
        string = string + "Off"
    else:
        string = string + "No"
    return string

def check_score(frame):
    left = model.predict(frame[309:325, 265:285].reshape(1, -1))
    right = model.predict(frame[309:325, 355:375].reshape(1, -1))
    return left, right

def caption(hit_type, left, right, update_left, update_right):
    caption = "None"
    if hit_type == "On-On":
        if update_left - left == 1 and update_right - right == 0:
            caption = "L"
        if update_left - left == 0 and update_right - right == 1:
            caption = "R"
        if update_left - left == 0 and update_right - right == 0:
            caption = "T"
    if hit_type == "On-Off":
        if update_left - left == 1 and update_right - right == 0:
            caption = "L"
        if update_left - left == 0 and update_right - right == 0:
            caption = "R"
    if hit_type == "Off-On":
        if update_left - left == 0 and update_right - right == 1:
            caption = "R"
        if update_left - left == 0 and update_right - right == 0:
            caption = "L"
    return caption

for i in os.listdir(os.getcwd() + "/videos"):
    if i.endswith(".mp4"): 
        match_number = int(i.split("-")[0])
        hit_number = int(i.split("-")[1].replace(".mp4", ""))
        cap = cv2.VideoCapture("videos/" + i)
        cap_end_point = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
        cap.set(1, cap_end_point - 1)  
        ret, frame = cap.read()
        hit_type = check_lights(frame)
        left, right = check_score(frame)
        cap.release()

        if hit_type == "On-On" or hit_type == "On-Off" or hit_type == "Off-On":
            next_hit = "videos/" + str(match_number) + "-" + str(hit_number + 1) + ".mp4"
            if os.path.isfile(next_hit) == True:
                cap = cv2.VideoCapture(next_hit)
                cap.set(1, 0)
                ret, frame = cap.read()
                update_left, update_right = check_score(frame)
                cap.release()
                priority = caption(hit_type, left, right, update_left, update_right)
                if priority != 'None':
                    os.rename("videos/" + i, "training_quarantine/" + priority + i)
        continue
    else:
        continue
