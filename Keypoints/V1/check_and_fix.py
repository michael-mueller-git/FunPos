from __future__ import annotations
import os
import sys

from torch.utils.data import DataLoader

sys.path.append(os.path.join(os.path.dirname(__file__), "lib", "keypoint_rcnn_training"))

from utils import collate_fn

from dataset import ClassDataset, train_transform

def test():
    for x in ["data/train", "data/test"]:
        dataset_test = ClassDataset(x, transform=train_transform(), demo=True)
        data_loader_test = DataLoader(dataset_test, batch_size=1, shuffle=False, collate_fn=collate_fn)
        print("check", x)
        iterator = iter(data_loader_test)
        f1 = dataset_test.imgs_files
        f2 = dataset_test.annotations_files
        i = -1
        while True:
            try:
                i += 1
                _ = next(iterator)
                if i % 100 == 0:
                    print(dataset_test.last_idx)
            except StopIteration:
                break
            except Exception as ex:
                print(ex)
                print("del", f1[dataset_test.last_idx])
                os.remove(os.path.join(x, "images", f1[dataset_test.last_idx]))
                os.remove(os.path.join(x, "annotations", f2[dataset_test.last_idx]))


if __name__ == "__main__":
    test()
