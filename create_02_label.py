from lib.funscript import Funscript
from lib.helper import get_videos
import os
import json

from scipy.interpolate import interp1d


DEBUG = False


def millisec_to_frame(milliseconds: int, fps: float) -> int:
    if milliseconds < 0: return 0
    return int(round(float(milliseconds)/(float(1000)/float(fps))))


for video_file in get_videos():
    print("video_file: " + video_file)
    fs_file = "".join(video_file[:-4] + ".funscript")
    param_file = "".join(video_file[:-4] + ".param")
    if not os.path.exists(fs_file):
        print("ERROR: Funscript not exists")
        continue
    if not os.path.exists(param_file):
        print("ERROR: Param not exists")
        continue

    with open(param_file, "r") as f:
        param = json.load(f)

    script, _ = Funscript.load(video_file, fs_file)
    actions = script.get_actions()

    mapped_actions = {}
    for action in actions:
        mapped_actions[millisec_to_frame(action['at'], param['fps'])] = action['pos']

    x = [k for k in mapped_actions.keys()]
    y = [mapped_actions[k] for k in mapped_actions.keys()]

    fx0 = interp1d(x, y, kind = 'quadratic')

    labels = {}
    for i in range(min(x), max(x)):
        labels[i] = float(fx0(i)) / 100.0

    if DEBUG:
        import matplotlib.pyplot as plt
        plt.plot([k for k in labels.keys()], [v for v in labels.values()])
        plt.show()

    with open("".join(video_file[:-4]) + '.labels', "w") as f:
        json.dump(labels, f)
        print("save labels")
