import os
import config

def get_videos() -> list:
    all_videos = []
    for video_file in os.listdir(config.train_dir):
        if not video_file.lower().endswith((".mkv", ".mp4")):
            continue
        video_file = os.path.join(config.train_dir, video_file)
        all_videos.append(video_file)

    return all_videos
