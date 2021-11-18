import posix
import cv2
import os
import copy
import json
import torch
import time
import config
import einops

from torch.utils.data import Dataset
from lib.ffmpegstream import FFmpegStream

import numpy as np


class Funscript_Dataset(Dataset):

    def __init__(self, data_dir):
        self.videos = [os.path.join(data_dir, f) \
                for f in os.listdir(data_dir) \
                if f.lower().endswith((".mp4", ".mkv"))
        ]
        self.video_idx = -1
        self.frame_num = 0
        self.last_label = 0
        self.stream = None
        self.params = {}
        self.labels = {}
        self.last_frames = []
        self.determine_len()


    def determine_len(self):
        self.length = 0
        for v in self.videos:
            with open("".join(v[:-4]) + ".labels", "r") as f:
                labels = json.load(f)
                self.length += int(
                    (
                        max([int(x) for x in labels.keys()])
                        - min([int(x) for x in labels.keys()])
                        - config.seq_len
                        - 1
                    ) / (
                        config.skip_frames + 1
                    )
                )


    def inc_frame_counter(self):
        self.frame_num += (config.skip_frames+1)


    def inc_video_idx(self):
        self.video_idx += 1
        if self.video_idx >= len(self.videos):
            self.video_idx = 0
        self.frame_num = 0


    def read_next_frame(self):
        while len(self.last_frames) >= config.seq_len:
            del self.last_frames[0]
        frame = self.stream.read()
        if frame is None:
            self.open_next_video()
        else:
            self.last_frames.append(frame)
            self.inc_frame_counter()


    def __len__(self):
        return self.length


    def load_next_video_frames(self):
        self.last_frames = []
        # skip frames without labels
        while self.frame_num < min(self.labels.keys()):
            self.read_next_frame()

        # fill buffer with given seq_len
        for _ in range(config.seq_len):
            self.read_next_frame()


    def open_next_video(self):
        self.inc_video_idx()
        with open("".join(self.videos[self.video_idx][:-4]) + '.param', "r") as f:
            self.param = json.load(f)
        with open("".join(self.videos[self.video_idx][:-4]) + '.labels', "r") as f:
            l = json.load(f)
            self.labels = {int(k):l[k] for k in l.keys()}
        self.param['resize'] = (config.img_width, config.img_height)
        if self.stream is not None:
            self.stream.stop()
        self.stream = FFmpegStream(
                self.videos[self.video_idx],
                self.param,
                skip_frames=config.skip_frames
        )
        self.load_next_video_frames()
        self.last_label = max(self.labels.keys())


    def __getitem__(self, idx):
        if idx == 0:
            self.video_idx = -1
            self.open_next_video()
        elif self.frame_num + config.skip_frames + 1 >= self.last_label:
            self.open_next_video()
        else:
            self.read_next_frame()

        frames = np.array(np.array([x - config.IMAGE_MEAN \
            for x in self.last_frames]))
        position = np.array([self.labels[self.frame_num-(i*(config.skip_frames+1))] \
            for i in reversed(range(config.seq_len))])

        frames = einops.rearrange(frames, 'time height width channel -> time channel height width')

        return torch.from_numpy(frames).float(), torch.from_numpy(position).float()
