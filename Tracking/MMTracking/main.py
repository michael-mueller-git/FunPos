import mmcv
import torch, torchvision
import mmdet
import mmtrack
import tempfile
from mmtrack.apis import inference_mot, init_model

print(mmcv.collect_env())

MODEL = 1

if MODEL == 1:
    mot_config = './mmtracking/configs/mot/deepsort/deepsort_faster-rcnn_fpn_4e_mot17-private-half.py'
    input_video = './mmtracking/demo/demo.mp4'
    imgs = mmcv.VideoReader(input_video)
    # build the model from a config file
    # mot_model = init_model(mot_config, device='cuda:0')
    mot_model = init_model(mot_config, device='cpu')
    prog_bar = mmcv.ProgressBar(len(imgs))
    out_dir = tempfile.TemporaryDirectory()
    out_path = out_dir.name
    # test and show/save the images
    for i, img in enumerate(imgs):
        result = inference_mot(mot_model, img, frame_id=i)
        mot_model.show_result(
                img,
                result,
                show=False,
                wait_time=int(1000. / imgs.fps),
                out_file=f'{out_path}/{i:06d}.jpg')
        prog_bar.update()

    output = './mot.mp4'
    print(f'\n making the output video at {output} with a FPS of {imgs.fps}')
    mmcv.frames2video(out_path, output, fps=imgs.fps, fourcc='mp4v')
    out_dir.cleanup()
    # run sot demo

if MODEL == 2:
    from mmtrack.apis import inference_sot
    sot_config = './mmtracking/configs/sot/siamese_rpn/siamese_rpn_r50_20e_lasot.py'
    sot_checkpoint = './checkpoints/siamese_rpn_r50_1x_lasot_20211203_151612-da4b3c66.pth'
    input_video = './mmtracking/demo/demo.mp4'
    # build the model from a config file and a checkpoint file
    # sot_model = init_model(sot_config, sot_checkpoint, device='cuda:0')
    sot_model = init_model(sot_config, sot_checkpoint, device='cpu')
    init_bbox = [371, 411, 450, 646]
    imgs = mmcv.VideoReader(input_video)
    prog_bar = mmcv.ProgressBar(len(imgs))
    out_dir = tempfile.TemporaryDirectory()
    out_path = out_dir.name
    for i, img in enumerate(imgs):
        result = inference_sot(sot_model, img, init_bbox, frame_id=i)
        sot_model.show_result(
                img,
                result,
                wait_time=int(1000. / imgs.fps),
                out_file=f'{out_path}/{i:06d}.jpg')
        prog_bar.update()
    output = './demo/sot.mp4'
    print(f'\n making the output video at {output} with a FPS of {imgs.fps}')
    mmcv.frames2video(out_path, output, fps=imgs.fps, fourcc='mp4v')
    out_dir.cleanup()
