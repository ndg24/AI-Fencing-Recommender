from pytube import YouTube
import os
import signal
import time
import traceback

class Timeout():
    class Timeout(Exception):
        pass
 
    def __init__(self, sec):
        self.sec = sec
 
    def __enter__(self):
        signal.signal(signal.SIGALRM, self.raise_timeout)
        signal.alarm(self.sec)
 
    def __exit__(self, *args):
        signal.alarm(0)
 
    def raise_timeout(self, *args):
        raise Timeout.Timeout()

directories = ['precut', 'videos', 'training_quarantine', 'more_training_data', 'final_training_clips', 'optical_flow', 'preinception_data', 'final_training_data', 'training_data']
for dirs in directories:
    if not os.path.exists(dirs):
        os.makedirs(dirs)

text_file = open("sabre_videos.txt", "r")
vids = text_file.read().split('\r')
print("First 3 links:", vids[:3])
text_file.close()

counter = 0
vids = vids[counter:]
for i in vids:
    try:
        with Timeout(600):
            start = time.time()
            yt = YouTube(i)
            yt.set_filename(str(counter))
            video = yt.get('mp4', '360p')
            video.download(os.getcwd() + '/precut/')
            print("Downloaded: ", i, "   ", (time.time() - start), "s")
    except:
        traceback.print_exc()
        print("Failed -", i)
    counter = counter + 1
