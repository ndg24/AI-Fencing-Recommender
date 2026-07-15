import os
import subprocess as sp

import cv2
import imageio_ffmpeg

FFMPEG_BIN = imageio_ffmpeg.get_ffmpeg_exe()
fps = str(13)

os.makedirs("training_quarantine", exist_ok=True)
os.makedirs("more_training_data", exist_ok=True)

for i in os.listdir(os.getcwd() + "/training_quarantine"):
    if i.endswith(".mp4"):
        cap = cv2.VideoCapture("training_quarantine/" + str(i))

        if i[0] == "L":
            i = "R" + i.lstrip("L")
        elif i[0] == "R":
            i = "L" + i.lstrip("R")

        output_file = "more_training_data/" + str(i).replace(".mp4", "-flipped") + ".mp4"
        cap.set(cv2.CAP_PROP_FPS, 10000)

        command = [
            FFMPEG_BIN,
            "-y",
            "-f", "rawvideo",
            "-vcodec", "rawvideo",
            "-s", "640x360",
            "-pix_fmt", "bgr24",
            "-r", fps,
            "-i", "-",
            "-an",
            "-vcodec", "mpeg4",
            "-b:v", "5000k",
            output_file,
        ]

        proc = sp.Popen(command, stdin=sp.PIPE, stderr=sp.PIPE)

        counter = 0
        while cap.isOpened():
            ret, frame = cap.read()
            counter = counter + 1
            if ret == True:
                frame = cv2.flip(frame, 1)
                proc.stdin.write(frame.tostring())
            else:
                break

        proc.stdin.close()
        proc.stderr.close()
        print("successful")

        cap.release()
