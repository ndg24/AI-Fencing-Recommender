import os
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError

import yt_dlp

directories = ['precut', 'videos', 'training_quarantine', 'more_training_data', 'final_training_clips', 'optical_flow', 'preinception_data', 'final_training_data', 'training_data']


def download_video(url, counter):
    output_template = os.path.join(os.getcwd(), 'precut', str(counter) + '.%(ext)s')
    ydl_opts = {
        'format': 'best[height<=360][ext=mp4]/best[ext=mp4]/best',
        'outtmpl': output_template,
        'merge_output_format': 'mp4',
        'quiet': True,
        'no_warnings': True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])


def main():
    for dirs in directories:
        if not os.path.exists(dirs):
            os.makedirs(dirs)

    with open("video_of_sabre.txt", "r") as text_file:
        vids = [line.strip() for line in text_file.read().splitlines() if line.strip()]
    print("First 3 links:", vids[:3])

    counter = 0
    with ThreadPoolExecutor(max_workers=1) as executor:
        for i in vids:
            try:
                start = time.time()
                future = executor.submit(download_video, i, counter)
                future.result(timeout=600)
                print("Downloaded: ", i, "   ", (time.time() - start), "s")
            except FutureTimeoutError:
                print("Timed out -", i)
            except Exception:
                traceback.print_exc()
                print("Failed -", i)
            counter = counter + 1


if __name__ == "__main__":
    main()
