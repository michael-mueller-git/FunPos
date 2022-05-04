import os
import json
import torch
import einops

from torch.utils.data import Dataset
from utils.ffmpegstream import FFmpegStream

import numpy as np


class Funscript_Dataset(Dataset):

    def __init__(self, data_dir, skip_frames, seq_len, img_width, img_height, output='time channel height width'):
        self.videos = [os.path.join(data_dir, f) \
                for f in os.listdir(data_dir) \
                if f.lower().endswith((".mp4", ".mkv"))
        ]
        self.skip_frames = skip_frames
        self.seq_len = seq_len
        self.img_width = img_width
        self.img_height = img_height
        self.output = output
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
                        - self.seq_len
                        - 1
                    ) / (
                        self.skip_frames + 1
                    )
                )


    def inc_frame_counter(self):
        self.frame_num += (self.skip_frames+1)


    def inc_video_idx(self):
        self.video_idx += 1
        if self.video_idx >= len(self.videos):
            self.video_idx = 0
        self.frame_num = 0


    def read_next_frame(self):
        while len(self.last_frames) >= self.seq_len:
            del self.last_frames[0]
        frame = self.stream.read()
        # frame = frame / 255.0
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
        for _ in range(self.seq_len):
            self.read_next_frame()


    def get_uniform_random_int(self, lower, upper):
        return torch.randint(lower, upper+1, (1,)).numpy()[0]


    def open_next_video(self):
        self.inc_video_idx()
        with open("".join(self.videos[self.video_idx][:-4]) + '.param', "r") as f:
            self.param = json.load(f)
        with open("".join(self.videos[self.video_idx][:-4]) + '.labels', "r") as f:
            l = json.load(f)
            self.labels = {int(k):l[k] for k in l.keys()}
        self.param['zoom'] = [
            self.param['zoom'][0] + self.get_uniform_random_int(-10, 10),
            self.param['zoom'][1] + self.get_uniform_random_int(-10, 10),
            self.param['zoom'][2] + self.get_uniform_random_int(-10, 10),
            self.param['zoom'][3] + self.get_uniform_random_int(-10, 10)
        ]
        self.param['resize'] = (self.img_width, self.img_height)
        if self.stream is not None:
            self.stream.stop()
        self.stream = FFmpegStream(
                self.videos[self.video_idx],
                self.param,
                skip_frames=self.skip_frames
        )
        self.load_next_video_frames()
        self.last_label = max(self.labels.keys())


    def __getitem__(self, idx):
        if idx == 0:
            self.video_idx = -1
            self.open_next_video()
        elif self.frame_num + self.skip_frames + 1 >= self.last_label:
            self.open_next_video()
        else:
            self.read_next_frame()

        frames = np.array(np.array(self.last_frames))
        position = np.array([self.labels[self.frame_num-(i*(self.skip_frames+1))] \
            for i in reversed(range(self.seq_len))])

        frames = einops.rearrange(frames, 'time height width channel -> ' + self.output)

        return torch.from_numpy(frames).float(), torch.from_numpy(position).float()
