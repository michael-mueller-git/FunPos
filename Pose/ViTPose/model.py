from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import cv2
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, 'ViTPose/')

from mmdet.apis import inference_detector, init_detector
from mmpose.apis import (inference_top_down_pose_model, init_pose_model,
                         process_mmdet_results, vis_pose_result)


class DetModel:
    MODEL_DICT = {
        'YOLOX-tiny': {
            'config':
            'mmdet_configs/configs/yolox/yolox_tiny_8x8_300e_coco.py',
            'model':
            'https://download.openmmlab.com/mmdetection/v2.0/yolox/yolox_tiny_8x8_300e_coco/yolox_tiny_8x8_300e_coco_20211124_171234-b4047906.pth',
        },
        'YOLOX-s': {
            'config':
            'mmdet_configs/configs/yolox/yolox_s_8x8_300e_coco.py',
            'model':
            'https://download.openmmlab.com/mmdetection/v2.0/yolox/yolox_s_8x8_300e_coco/yolox_s_8x8_300e_coco_20211121_095711-4592a793.pth',
        },
        'YOLOX-l': {
            'config':
            'mmdet_configs/configs/yolox/yolox_l_8x8_300e_coco.py',
            'model':
            'https://download.openmmlab.com/mmdetection/v2.0/yolox/yolox_l_8x8_300e_coco/yolox_l_8x8_300e_coco_20211126_140236-d3bd2b23.pth',
        },
        'YOLOX-x': {
            'config':
            'mmdet_configs/configs/yolox/yolox_x_8x8_300e_coco.py',
            'model':
            'https://download.openmmlab.com/mmdetection/v2.0/yolox/yolox_x_8x8_300e_coco/yolox_x_8x8_300e_coco_20211126_140254-1ef88d67.pth',
        },
    }

    def __init__(self, device: str | torch.device, model_name: str = 'YOLOX-l'):
        self.device = torch.device(device)
        # self._load_all_models_once()
        self.model_name = ''
        self.model = self._load_model(model_name)

    def _load_all_models_once(self) -> None:
        for name in self.MODEL_DICT:
            self._load_model(name)

    def _load_model(self, name: str) -> nn.Module:
        dic = self.MODEL_DICT[name]
        return init_detector(dic['config'], dic['model'], device=self.device)

    def set_model(self, name: str) -> None:
        if name == self.model_name:
            return
        self.model_name = name
        self.model = self._load_model(name)

    def detect_and_visualize(
            self, image: np.ndarray,
            score_threshold: float) -> tuple[list[np.ndarray], np.ndarray]:
        out = self.detect(image)
        vis = self.visualize_detection_results(image, out, score_threshold)
        return out, vis

    def detect(self, image: np.ndarray) -> list[np.ndarray]:
        image = image[:, :, ::-1]  # RGB -> BGR
        out = inference_detector(self.model, image)
        return out

    def visualize_detection_results(
            self,
            image: np.ndarray,
            detection_results: list[np.ndarray],
            score_threshold: float = 0.3) -> np.ndarray:
        person_det = [detection_results[0]] + [np.array([]).reshape(0, 5)] * 79

        image = image[:, :, ::-1]  # RGB -> BGR
        vis = self.model.show_result(image,
                                     person_det,
                                     score_thr=score_threshold,
                                     bbox_color=None,
                                     text_color=(200, 200, 200),
                                     mask_color=None)
        return vis[:, :, ::-1]  # BGR -> RGB


class PoseModel:
    MODEL_DICT = {
        'ViTPose-B (single-task train)': {
            'config': 'ViTPose/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/ViTPose_base_coco_256x192.py',
            'model': 'models/vitpose-b.pth',
        },
        'ViTPose-L (single-task train)': {
            'config': 'ViTPose/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/ViTPose_large_coco_256x192.py',
            'model': 'models/vitpose-l.pth',
        },
        'ViTPose-H (single-task train)': {
            'config': 'ViTPose/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/ViTPose_huge_coco_256x192.py',
            'model': 'models/vitpose-h.pth',
        },
        'ViTPose-B (multi-task train, COCO)': {
            'config': 'ViTPose/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/ViTPose_base_coco_256x192.py',
            'model': 'models/vitpose-b-multi-coco.pth',
        },
        'ViTPose-L (multi-task train, COCO)': {
            'config': 'ViTPose/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/ViTPose_large_coco_256x192.py',
            'model': 'models/vitpose-l-multi-coco.pth',
        },
        'ViTPose-H (multi-task train, COCO)': {
            'config': 'ViTPose/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/ViTPose_huge_coco_256x192.py',
            'model': 'models/vitpose-h-multi-coco.pth',
        },
    }

    def __init__(self, device: str | torch.device, model_name: str = 'ViTPose-B (multi-task train, COCO)'):
        self.device = torch.device(device)
        self.model_name = ''
        self.model = self._load_model(model_name)

    def _load_all_models_once(self) -> None:
        for name in self.MODEL_DICT:
            self._load_model(name)

    def _load_model(self, name: str) -> nn.Module:
        dic = self.MODEL_DICT[name]
        if not os.path.exists(dic['model']):
            print('TODO download', dic['model'])
        ckpt_path = dic['model']
        model = init_pose_model(dic['config'], ckpt_path, device=self.device)
        return model

    def set_model(self, name: str) -> None:
        if name == self.model_name:
            return
        self.model_name = name
        self.model = self._load_model(name)

    def predict_pose_and_visualize(
        self,
        image: np.ndarray,
        det_results: list[np.ndarray],
        box_score_threshold: float,
        kpt_score_threshold: float,
        vis_dot_radius: int,
        vis_line_thickness: int,
    ) -> tuple[list[dict[str, np.ndarray]], np.ndarray]:
        out = self.predict_pose(image, det_results, box_score_threshold)
        vis = self.visualize_pose_results(image, out, kpt_score_threshold,
                                          vis_dot_radius, vis_line_thickness)
        return out, vis

    def predict_pose(
            self,
            image: np.ndarray,
            det_results: list[np.ndarray],
            box_score_threshold: float = 0.5) -> list[dict[str, np.ndarray]]:
        image = image[:, :, ::-1]  # RGB -> BGR
        person_results = process_mmdet_results(det_results, 1)
        out, _ = inference_top_down_pose_model(self.model,
                                               image,
                                               person_results=person_results,
                                               bbox_thr=box_score_threshold,
                                               format='xyxy')
        return out

    def visualize_pose_results(self,
                               image: np.ndarray,
                               pose_results: list[dict[str, np.ndarray]],
                               kpt_score_threshold: float = 0.3,
                               vis_dot_radius: int = 4,
                               vis_line_thickness: int = 1) -> np.ndarray:
        image = image[:, :, ::-1]  # RGB -> BGR
        vis = vis_pose_result(self.model,
                              image,
                              pose_results,
                              kpt_score_thr=kpt_score_threshold,
                              radius=vis_dot_radius,
                              thickness=vis_line_thickness)
        return vis[:, :, ::-1]  # BGR -> RGB


class AppModel:
    def __init__(self, device: str | torch.device):
        self.device = device
        self.det_model = None
        self.pose_model = None

    def init_models(self, det_model_name, pose_model_name):
        if self.det_model is None:
            self.det_model = DetModel(self.device, det_model_name)
        if self.pose_model is None:
            self.pose_model = PoseModel(self.device, pose_model_name)

    def run(
        self, video_path: str, det_model_name: str, pose_model_name: str,
        box_score_threshold: float, max_num_frames: int,
        kpt_score_threshold: float, vis_dot_radius: int,
        vis_line_thickness: int
    ) -> tuple[str, list[list[dict[str, np.ndarray]]]]:
        self.init_models(det_model_name, pose_model_name)
        if video_path is None:
            return
        self.det_model.set_model(det_model_name)
        self.pose_model.set_model(pose_model_name)

        cap = cv2.VideoCapture(video_path)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        fps = cap.get(cv2.CAP_PROP_FPS)

        preds_all = []

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        temp_file = tempfile.NamedTemporaryFile(suffix='.mp4')
        writer = cv2.VideoWriter(temp_file.name, fourcc, fps, (width, height))
        for _ in range(max_num_frames):
            ok, frame = cap.read()
            if not ok:
                break
            rgb_frame = frame[:, :, ::-1]
            det_preds = self.det_model.detect(rgb_frame)
            preds, vis = self.pose_model.predict_pose_and_visualize(
                rgb_frame, det_preds, box_score_threshold, kpt_score_threshold,
                vis_dot_radius, vis_line_thickness)
            preds_all.append(preds)
            writer.write(vis[:, :, ::-1])
        cap.release()
        writer.release()

        out_file = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)
        subprocess.run(
            f'ffmpeg -y -loglevel quiet -stats -i {temp_file.name} -c:v libx264 {out_file.name}'
            .split())
        return out_file.name, preds_all

    def visualize_pose_results(self, video_path: str,
                               pose_preds_all: list[list[dict[str,
                                                              np.ndarray]]],
                               kpt_score_threshold: float, vis_dot_radius: int,
                               vis_line_thickness: int) -> str:
        if video_path is None or pose_preds_all is None:
            return
        cap = cv2.VideoCapture(video_path)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        fps = cap.get(cv2.CAP_PROP_FPS)

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        temp_file = tempfile.NamedTemporaryFile(suffix='.mp4', delete=True)
        writer = cv2.VideoWriter(temp_file.name, fourcc, fps, (width, height))
        for pose_preds in pose_preds_all:
            ok, frame = cap.read()
            if not ok:
                break
            rgb_frame = frame[:, :, ::-1]
            vis = self.pose_model.visualize_pose_results(
                rgb_frame, pose_preds, kpt_score_threshold, vis_dot_radius,
                vis_line_thickness)
            writer.write(vis[:, :, ::-1])
        cap.release()
        writer.release()

        out_file = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)
        subprocess.run(
            f'ffmpeg -y -loglevel quiet -stats -i {temp_file.name} -c:v libx264 {out_file.name}'
            .split())
        return out_file.name
