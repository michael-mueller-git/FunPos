import os
import torch
import torchvision

from PIL import Image
from torch.utils.data import Dataset

import numpy as np


class SingleMaskDataset(Dataset):

    def __init__(self, data_dir, mask_filename_postfix='_anno'):
        self.data_dir = data_dir
        self.mask_filename_postfix = mask_filename_postfix
        self.transforms = [torchvision.transforms.ToTensor()]
        self.__load_database()


    def __load_database(self):
        self.database = [(os.path.join(self.data_dir, f), os.path.join(self.data_dir, f.replace('.png', '{}.png'.format(self.mask_filename_postfix)))) \
                for f in os.listdir(self.data_dir) \
                if f.endswith((".png")) \
                and not f.endswith(("{}.png".format(self.mask_filename_postfix))) \
                and os.path.exists(os.path.join(self.data_dir, f.replace('.png', '{}.png'.format(self.mask_filename_postfix))))
            ]


    def get_dataset_list(self) -> list[tuple]:
        return self.database


    def __len__(self):
        return len(self.database)


    def __getitem__(self, idx):
        idx = idx % len(self.database)
        img = Image.open(self.database[idx][0]).convert("RGB")
        mask = np.array(Image.open(self.database[idx][1]))

        obj_ids = np.unique(mask)
        obj_ids = obj_ids[1:]

        masks = mask == obj_ids[:, None, None]

        num_objs = len(obj_ids)
        boxes = []
        for i in range(num_objs):
          pos = np.where(masks[i])

          xmin = np.min(pos[1])
          xmax = np.max(pos[1])
          ymin = np.min(pos[0])
          ymax = np.max(pos[0])

          if abs((xmax-xmin) * (ymax-ymin)) < 5:
            obj_ids = np.delete(obj_ids, [i])
            continue

          boxes.append([xmin, ymin, xmax, ymax])

        num_objs = len(obj_ids)
        boxes = torch.as_tensor(boxes, dtype=torch.float32)

        labels = torch.ones((num_objs,), dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        for i in self.transforms:
            img = i(img)

        target = {
            "boxes": boxes,
            "labels": labels,
            "masks": masks
        }

        return img.double(), target


if __name__ == '__main__':
    dataset = SingleMaskDataset('./data/train')
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
    print(dataset.get_dataset_list())

