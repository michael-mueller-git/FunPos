import os
import shutil
import torch

from model import AppModel

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_video = 'data/test.mp4'
    detector_name = 'YOLOX-l'
    pose_model_name = 'ViTPose-L (single-task train)'
    det_score_threshold = 0.5
    max_num_frames = 300
    vis_kpt_score_threshold = 0.3
    vis_dot_radius = 4
    vis_line_thickness = 2

    model = AppModel(device=device)
    out_name, pose_preds = model.run(
             input_video,
             detector_name,
             pose_model_name,
             det_score_threshold,
             max_num_frames,
             vis_kpt_score_threshold,
             vis_dot_radius,
             vis_line_thickness,
         )

    shutil.move(out_name, os.getcwd() + os.sep + os.path.basename(out_name))
