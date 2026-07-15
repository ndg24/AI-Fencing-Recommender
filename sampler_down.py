import os
import subprocess as sp

import cv2
import imageio_ffmpeg

FFMPEG_BIN = imageio_ffmpeg.get_ffmpeg_exe()
fps = str(13)
downsample_until_frame_number = 16
downsample_by_divisor = 2

os.makedirs("training_data", exist_ok=True)
os.makedirs("training_quarantine", exist_ok=True)

for i in os.listdir(os.getcwd() + "/training_data"):
    if i.endswith(".mp4"):
        cap = cv2.VideoCapture("training_data/" + str(i))
        output_file = "training_quarantine/" + str(i)
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
                if counter <= downsample_until_frame_number and counter % downsample_by_divisor == 0:
                    proc.stdin.write(frame.tostring())
                elif counter > downsample_until_frame_number:
                    proc.stdin.write(frame.tostring())
            else:
                break
        proc.stdin.close()
        proc.stderr.close()
        print(i + "-successful")
        cap.release()
