import torch
import numpy as np
from abc import ABCMeta, abstractmethod


class Anchor(metaclass=ABCMeta):
    def __init__(self, model_name='yolo'):
        self.model_name = model_name.lower()
        assert model_name in ['yolo', 'ssd', 'retina', 'yolov3']

    @abstractmethod
    def create_anchors(self):
        pass


class YOLOv3Anchor(Anchor):
    def __init__(self):
        super().__init__()
        self.anchor_whs = {"small": [(10, 13), (16, 30), (33, 23)],
                           "middle": [(30, 61), (62, 45), (59, 119)],
                           "large": [(116, 90), (156, 198), (373, 326)]}
        self.center_anchors = self.create_anchors()

    def to_device(self, device):
        self.center_anchors[0] = self.center_anchors[0].to(device)
        self.center_anchors[1] = self.center_anchors[1].to(device)
        self.center_anchors[2] = self.center_anchors[2].to(device)

    def anchor_for_scale(self, grid_size, wh):

        center_anchors = []
        for y in range(grid_size):
            for x in range(grid_size):
                cx = x + 0.5
                cy = y + 0.5
                for anchor_wh in wh:
                    w = anchor_wh[0]
                    h = anchor_wh[1]
                    center_anchors.append([cx, cy, w, h])

        print('done!')
        center_anchors_numpy = np.array(center_anchors).astype(np.float32)                          # to numpy  [845, 4]
        center_anchors_tensor = torch.from_numpy(center_anchors_numpy)                              # to tensor [845, 4]
        center_anchors_tensor = center_anchors_tensor.view(grid_size, grid_size, 3, 4)              # [13, 13, 5, 4]
        return center_anchors_tensor

    def create_anchors(self):

        print('make yolo anchor...')

        wh_large = torch.from_numpy(np.array(self.anchor_whs["large"]) / 32)    # 416 / 32 = 13 - large feature
        wh_middle = torch.from_numpy(np.array(self.anchor_whs["middle"]) / 16)   # 416 / 16 = 26 - middle feature
        wh_small = torch.from_numpy(np.array(self.anchor_whs["small"]) / 8)     # 416 / 8 = 52  - small feature

        center_anchors_large = self.anchor_for_scale(13, wh_large)
        center_anchors_middle = self.anchor_for_scale(26, wh_middle)
        center_anchors_small = self.anchor_for_scale(52, wh_small)

        return [center_anchors_large, center_anchors_middle, center_anchors_small]

