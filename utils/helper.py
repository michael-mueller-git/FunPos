import os
from utils.config import CONFIG

def get_videos() -> list:
    all_videos = []
    for video_file in os.listdir(CONFIG['general']['train_dir']):
        if not video_file.lower().endswith((".mkv", ".mp4")):
            continue
        video_file = os.path.join(CONFIG['general']['train_dir'], video_file)
        all_videos.append(video_file)

    return all_videos
