from lib.ffmpegstream import FFmpegStream
from lib.vrprojection import VrProjection
from lib.helper import get_videos
import cv2
import os
import json

for video_file in get_videos():
    print("video:", video_file)
    if os.path.exists("".join(video_file[:-4]) + '.param'):
        print("skip, parame exist")
        continue

    while True:
        projection = VrProjection(video_file)
        config = projection.get_parameter()
        video = FFmpegStream(video_file, config)

        while video.isOpen():
            frame = video.read()
            if frame is None: break
            cv2.imshow('Frame', frame)
            if cv2.waitKey(3) == ord('q'): break
        video.stop()
        cv2.destroyAllWindows()

        selection = input("parameter ok? -> save parameters? [y/N] ")
        if selection.lower() == "y":
            print('save param')
            with open("".join(video_file[:-4]) + '.param', "w") as f:
                json.dump(config, f)
                break
