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

FFMPEG_BIN = "ffmpeg"

import cPickle
with open('logistic_classifier_0-15.pkl', 'rb') as fid:
    model = cPickle.load(fid)

fps = str(13)
jump_length = 260
hide_length = 200
video_number = 0

videos_to_cut = 0
for i in os.listdir(os.getcwd() + "/precut"):
    if i.endswith(".mp4"):
        videos_to_cut = videos_to_cut + 1

already_processed = 400
for vid in os.listdir(os.getcwd() + "/precut"):
    if vid.endswith(".mp4") and int(vid.replace(".mp4", "")) >= already_processed:
        clips_recorded = 0
        recording_mode = False
    
        cap = cv2.VideoCapture("precut/" + str(vid))
        
        cap_end_point = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
        cap.release()
        cap_end_point = cap_end_point - jump_length
        position = 2000

        while position < cap_end_point:
            cap = cv2.VideoCapture("precut/" + str(vid))

            if position == cap_end_point:
                break

            if recording_mode == True:
                output_file = 'videos/' + str(vid).replace(".mp4", "") + "-" + str(clips_recorded) + '.mp4'
                
                command = [FFMPEG_BIN,
                '-y',
                '-f', 'rawvideo',
                '-vcodec', 'rawvideo',
                '-s', '640*360',
                '-pix_fmt', 'bgr24',
                '-r', fps,
                '-i', '-',
                '-an',
                '-vcodec', 'mpeg4',
                '-b:v', '5000k',
                output_file ]

                frames_till_video_end = jump_length
                proc = sp.Popen(command, stdin=sp.PIPE, stderr=sp.PIPE)
            
            if cap.isOpened():
                cap.set(1, position)
                cap.set(cv2.cv.CV_CAP_PROP_FPS, 10000)

                while cap.isOpened():
                    ret, frame = cap.read()
                    position = position + 1

                    if recording_mode == False:
                        if position == (cap_end_point):
                            break
                        elif cap.get(cv2.cv.CV_CAP_PROP_POS_FRAMES) >= cap_end_point:
                            position = cap.get(cv2.cv.CV_CAP_PROP_POS_FRAMES)
                            break
                        try:
                            if (np.sum(abs(frame[337:348, 234:250].astype(int) - white_box.astype(int))) <= 7000) or (np.sum(abs(frame[337:348, 390:406].astype(int) - white_box.astype(int))) <= 7000) or (np.sum(abs(frame[330:334, 380:500].astype(int) - green_box.astype(int))) <= 40000) or (np.sum(abs(frame[330:334, 140:260].astype(int) - red_box.astype(int))) <= 40000):
                                left_score = model.predict(frame[309:325, 265:285].reshape(1, -1))
                                right_score = model.predict(frame[309:325, 355:375].reshape(1, -1))
                                if (left_score == 15) or (right_score == 15):
                                    position = position + 25
                                    break
                                elif (left_score == 0) and (right_score == 0):
                                    position = position + 25
                                    break
                                else:
                                    position = position - 50
                                    recording_mode = True
                                    break
                        except:
                            break
                    
                    if recording_mode == True:
                        if frames_till_video_end >= hide_length:
                            if position % 2 == 0:
                                proc.stdin.write(frame.tostring())
                        
                        frames_till_video_end = frames_till_video_end - 1
                        if frames_till_video_end == 0:
                            recording_mode = False
                            proc.stdin.close()
                            proc.stderr.close()
                            clips_recorded = clips_recorded + 1
                            break   
            else:
                print("Failed to open video")

            cap.release()
            video_number = video_number + 1
