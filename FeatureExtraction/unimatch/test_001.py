import numpy as np
import torch
import cv2

from unimatch.unimatch import UniMatch
from utils.flow_viz import flow_to_image

model = UniMatch(feature_channels=128,
                 num_scales=1,
                 upsample_factor=8,
                 ffn_dim_expansion=4,
                 num_transformer_layers=6,
                 reg_refine=False,
                 task='flow')

model.eval()
# checkpoint_path = 'pretrained/gmflow-scale2-regrefine6-mixdata-train320x576-4e7b215d.pth'
checkpoint_path = './pretrained/gmflow-scale1-mixdata-train320x576-4c3a6e9a.pth'
checkpoint_flow = torch.load(checkpoint_path)
model.load_state_dict(checkpoint_flow['model'], strict=True)
attn_type = 'swin'
attn_splits_list = [2]
corr_radius_list = [-1]
prop_radius_list = [-1]
num_reg_refine = 1

cap = cv2.VideoCapture('demo/test.mkv')
prev = None

with torch.no_grad():
    while cap.isOpened():
        ret, img = cap.read()
        if not ret: break
        height, width = img.shape[:2]
        imgL = img[:, :int(width/2)]
        imgL = cv2.resize(imgL, (384, 384))
        if prev is None:
            prev = imgL
            continue

        image1 = np.array(prev).astype(np.float32)
        image2 = np.array(imgL).astype(np.float32)

        image1 = torch.from_numpy(image1).permute(2, 0, 1).float().unsqueeze(0)
        image2 = torch.from_numpy(image2).permute(2, 0, 1).float().unsqueeze(0)

        results_dict = model(image1, image2,
                             attn_type=attn_type,
                             attn_splits_list=attn_splits_list,
                             corr_radius_list=corr_radius_list,
                             prop_radius_list=prop_radius_list,
                             num_reg_refine=num_reg_refine,
                             task='flow',
                             )

        flow_pr = results_dict['flow_preds'][-1]  # [1, 2, H, W] or [1, H, W]
        flow = flow_pr[0].permute(1, 2, 0).cpu().numpy()  # [H, W, 2]

        u = flow[:, :, 0]
        v = flow[:, :, 1]
        rad = np.sqrt(u ** 2 + v ** 2)
        a = np.arctan2(-v, -u) / np.pi
        print(rad)
        print(a)


        output = flow_to_image(flow)  # [H, W, 3]

        prev = imgL

        cv2.imshow('flow', output)
        if cv2.waitKey(1) == ord('q'):
            break

cv2.destroyAllWindows()
cap.release()
