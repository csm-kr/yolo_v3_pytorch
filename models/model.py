import os
import wget
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from models.anchors import YOLOv3Anchor
from utils.util import bar_custom, cxcy_to_xy
from torchvision.ops.boxes import nms as torchvision_nms


class ResBlock(nn.Module):
    def __init__(self, in_channel):
        super().__init__()
        self.features = nn.Sequential(nn.Conv2d(in_channel, in_channel//2, 1, stride=1, padding=0, bias=False),
                                      nn.BatchNorm2d(in_channel//2, momentum=0.9),
                                      nn.LeakyReLU(0.1),
                                      nn.Conv2d(in_channel//2, in_channel, 3, stride=1, padding=1, bias=False),
                                      nn.BatchNorm2d(in_channel, momentum=0.9),
                                      nn.LeakyReLU(0.1),
                                      )

    def forward(self, x):
        residual = x
        x = self.features(x)
        x += residual
        return x


class Darknet53(nn.Module):

    '''
    num_params
    pytorch
    40584928

    darknet
    40620640
    '''

    def __init__(self, block=ResBlock, num_classes=1000, pretrained=True):
        super().__init__()

        self.num_classes = num_classes
        self.features1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32, momentum=0.9),
            nn.LeakyReLU(0.1),
            nn.Conv2d(32, 64, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64, momentum=0.9),
            nn.LeakyReLU(0.1),
            self.make_layer(block, in_channels=64, num_blocks=1),
            nn.Conv2d(64, 128, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128, momentum=0.9),
            nn.LeakyReLU(0.1),
            self.make_layer(block, in_channels=128, num_blocks=2),
            nn.Conv2d(128, 256, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256, momentum=0.9),
            nn.LeakyReLU(0.1),
            self.make_layer(block, in_channels=256, num_blocks=8),
        )

        self.features2 = nn.Sequential(
            nn.Conv2d(256, 512, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512, momentum=0.9),
            nn.LeakyReLU(0.1),
            self.make_layer(block, in_channels=512, num_blocks=8),
        )

        self.features3 = nn.Sequential(
            nn.Conv2d(512, 1024, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(1024, momentum=0.9),
            nn.LeakyReLU(0.1),
            self.make_layer(block, in_channels=1024, num_blocks=4),
        )

        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(1024, self.num_classes)
        self.init_layer()

        if pretrained:

            self.darknet53_url = "https://pjreddie.com/media/files/darknet53.conv.74"
            file = "darknet53.pth"
            directory = os.path.join(torch.hub.get_dir(), 'checkpoints')
            os.makedirs(directory, exist_ok=True)
            pretrained_file = os.path.join(directory, file)
            if os.path.exists(pretrained_file):
                print("weight already exist...!")
            else:
                print("Download darknet53 pre-trained weight...")
                wget.download(url=self.darknet53_url, out=pretrained_file, bar=bar_custom)
                print('')
                print("Done...!")
            self.load_darknet_weights(pretrained_file)


    def init_layer(self):
        for layer in self.children():
            if isinstance(layer, nn.Conv2d):
                nn.init.xavier_uniform_(layer.weight)

    def forward(self, x):
        x = self.features1(x)
        x = self.features2(x)
        x = self.features3(x)
        x = self.gap(x)
        x = x.view(-1, 1024)
        x = self.fc(x)
        return x

    def make_layer(self, block, in_channels, num_blocks):
        layers = []
        for i in range(0, num_blocks):
            layers.append(block(in_channels))
        return nn.Sequential(*layers)

    def load_darknet_weights(self, weights_path):
        """Parses and loads the weights stored in 'weights_path'"""

        # Open the weights file
        with open(weights_path, "rb") as f:
            header = np.fromfile(f, dtype=np.int32, count=5)  # First five are header values
            self.header_info = header  # Needed to write header when saving weights
            self.seen = header[3]  # number of images seen during training
            weights = np.fromfile(f, dtype=np.float32)  # The rest are weights

        ptr = 0
        conv_layer = None
        # refer to https://github.com/eriklindernoren/PyTorch-YOLOv3/blob/master/models.py
        for i, module in enumerate(self.modules()):
            if isinstance(module, nn.Conv2d):
                conv_layer = module
            if isinstance(module, nn.BatchNorm2d):
                bn_layer = module
                num_b = bn_layer.bias.numel()  # Number of biases

                # Bias
                bn_b = torch.from_numpy(weights[ptr: ptr + num_b]).view_as(bn_layer.bias)
                bn_layer.bias.data.copy_(bn_b)
                ptr += num_b

                # Weight
                bn_w = torch.from_numpy(weights[ptr: ptr + num_b]).view_as(bn_layer.weight)
                bn_layer.weight.data.copy_(bn_w)
                ptr += num_b

                # Running Mean
                bn_rm = torch.from_numpy(weights[ptr: ptr + num_b]).view_as(bn_layer.running_mean)
                bn_layer.running_mean.data.copy_(bn_rm)
                ptr += num_b

                # Running Var
                bn_rv = torch.from_numpy(weights[ptr: ptr + num_b]).view_as(bn_layer.running_var)
                bn_layer.running_var.data.copy_(bn_rv)
                ptr += num_b
            else:
                continue
            # Load conv. weights
            num_w = conv_layer.weight.numel()
            conv_w = torch.from_numpy(weights[ptr: ptr + num_w]).view_as(conv_layer.weight)
            conv_layer.weight.data.copy_(conv_w)
            ptr += num_w


class YoloV3(nn.Module):
    def __init__(self, baseline, num_classes=80):
        super().__init__()
        self.baseline = baseline
        self.anchor = YOLOv3Anchor()
        self.anchor_number = 3
        self.num_classes = num_classes

        self.extra_conv1 = nn.Sequential(
            nn.Conv2d(384, 128, 1, stride=1, padding=0),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(128, 256, 3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, inplace=True),

            nn.Conv2d(256, 128, 1, stride=1, padding=0),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(128, 256, 3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, inplace=True),

            nn.Conv2d(256, 128, 1, stride=1, padding=0),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(128, 256, 3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self.pred_1 = nn.Conv2d(256, self.anchor_number * (1 + 4 + self.num_classes), 1)

        self.extra_conv2 = nn.Sequential(
            nn.Conv2d(768, 256, 1, stride=1, padding=0),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(256, 512, 3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1, inplace=True),

            nn.Conv2d(512, 256, 1, stride=1, padding=0),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(256, 512, 3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1, inplace=True),

            nn.Conv2d(512, 256, 1, stride=1, padding=0),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(256, 512, 3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self.pred_2 = nn.Conv2d(512, self.anchor_number * (1 + 4 + self.num_classes), 1)
        self.conv_128x1x1 = nn.Sequential(
            nn.Conv2d(512, 128, 1, stride=1, padding=0),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, inplace=True),
        )

        self.extra_conv3 = nn.Sequential(
            nn.Conv2d(1024, 512, 1, stride=1, padding=0),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(512, 1024, 3, stride=1, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1, inplace=True),

            nn.Conv2d(1024, 512, 1, stride=1, padding=0),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(512, 1024, 3, stride=1, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1, inplace=True),

            nn.Conv2d(1024, 512, 1, stride=1, padding=0),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(512, 1024, 3, stride=1, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self.pred_3 = nn.Conv2d(1024, self.anchor_number * (1 + 4 + self.num_classes), 1)
        self.conv_256x1x1 = nn.Sequential(
            nn.Conv2d(1024, 256, 1, stride=1, padding=0),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.initialize()
        print("num_params : ", self.count_parameters())

    def initialize(self):
        def init_layer(layer):
            if isinstance(layer, nn.Conv2d):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, val=0)

        self.extra_conv1.apply(init_layer)
        self.extra_conv2.apply(init_layer)
        self.extra_conv3.apply(init_layer)
        self.pred_1.apply(init_layer)
        self.pred_2.apply(init_layer)
        self.pred_3.apply(init_layer)
        self.conv_128x1x1.apply(init_layer)
        self.conv_256x1x1.apply(init_layer)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x):

        p1 = x = self.baseline.features1(x)  # after residual3 [B, 256, 52, 52]
        p2 = x = self.baseline.features2(x)  # after residual4 [B, 512, 26, 26]
        p3 = x = self.baseline.features3(x)  # after residual5 [B, 1024, 13, 13]

        p3_up = self.conv_256x1x1(p3)                        # [B, 256, 13, 13]
        p3_up = F.interpolate(p3_up, scale_factor=2.)        # [B, 256, 26, 26]
        p3 = self.extra_conv3(p3)                            # [B, 1024, 13, 13]
        p3 = self.pred_3(p3)                                 # [B, 255, 13, 13] ***

        p2 = torch.cat([p2, p3_up], dim=1)                   # [B, 768, 26, 26]
        p2 = self.extra_conv2(p2)                            # [B, 512, 26, 26]

        p2_up = self.conv_128x1x1(p2)                        # [B, 128, 26, 26]
        p2_up = F.interpolate(p2_up, scale_factor=2.)        # [B, 128, 52, 52]
        p2 = self.pred_2(p2)                                 # [B, 255, 26, 26] ***

        p1 = torch.cat([p1, p2_up], dim=1)                   # [B, 384, 52, 52]
        p1 = self.extra_conv1(p1)                            # [B, 256, 52, 52]
        p1 = self.pred_1(p1)                                 # [B, 255, 52, 52] ***

        p_s = p1.permute(0, 2, 3, 1)  # B, 52, 52, 255
        p_m = p2.permute(0, 2, 3, 1)  # B, 26, 26, 255
        p_l = p3.permute(0, 2, 3, 1)  # B, 13, 13, 255

        return [p_l, p_m, p_s]

    def pred2target(self, pred):
        # pred to target
        out_size = pred.size(1)  # 13, 13
        pred_targets = pred.view(-1, out_size, out_size, 3, 5 + self.num_classes)
        pred_target_xy = pred_targets[..., :2].sigmoid()  # 0, 1 sigmoid(tx, ty) -> bx, by
        pred_target_wh = pred_targets[..., 2:4]  # 2, 3
        pred_objectness = pred_targets[..., 4].unsqueeze(-1).sigmoid()  # 4        class probability
        pred_classes = pred_targets[..., 5:].sigmoid()  # 20 / 80  classes
        return pred_target_xy, pred_target_wh, pred_objectness, pred_classes

    def decodeYoloV3(self, pred_targets, center_anchor):

        # pred to target
        out_size = pred_targets.size(1)  # 13, 13
        pred_txty, pred_twth, pred_objectness, pred_classes = self.pred2target(pred_targets)

        # decode
        center_anchors_xy = center_anchor[..., :2]  # torch.Size([13, 13, 3, 2])
        center_anchors_wh = center_anchor[..., 2:]  # torch.Size([13, 13, 3, 2])

        pred_bbox_xy = center_anchors_xy.floor().expand_as(pred_txty) + pred_txty
        pred_bbox_wh = center_anchors_wh.expand_as(pred_twth) * pred_twth.exp()
        pred_bbox = torch.cat([pred_bbox_xy, pred_bbox_wh], dim=-1)                     # [B, 13, 13, 3, 4]
        pred_bbox = pred_bbox.view(-1, out_size * out_size * 3, 4) / out_size           # [B, 507, 4]
        pred_cls = pred_classes.reshape(-1, out_size * out_size * 3, self.num_classes)  # [B, 507, 80]
        pred_conf = pred_objectness.reshape(-1, out_size * out_size * 3)
        return pred_bbox, pred_cls, pred_conf

    def predict(self, preds, list_center_anchors, opts):
        pred_targets_l, pred_targets_m, pred_targets_s = preds
        center_anchor_l, center_anchor_m, center_anchor_s = list_center_anchors

        pred_bbox_l, pred_cls_l, pred_conf_l = self.decodeYoloV3(pred_targets_l, center_anchor_l)
        pred_bbox_m, pred_cls_m, pred_conf_m = self.decodeYoloV3(pred_targets_m, center_anchor_m)
        pred_bbox_s, pred_cls_s, pred_conf_s = self.decodeYoloV3(pred_targets_s, center_anchor_s)

        pred_bbox = torch.cat([pred_bbox_l, pred_bbox_m, pred_bbox_s], dim=1)                # [B, 10647, 4]
        pred_cls = torch.cat([pred_cls_l, pred_cls_m, pred_cls_s], dim=1)                    # [B, 10647, 80]
        pred_conf = torch.cat([pred_conf_l, pred_conf_m, pred_conf_s], dim=1)

        scores = (pred_cls * pred_conf.unsqueeze(-1))          # [B, 10647, 21]
        boxes = cxcy_to_xy(pred_bbox).clamp(0, 1)              # [B, 10647, 4]

        bbox, label, score = self._suppress(boxes, scores, opts)
        return bbox, label, score

    def _suppress(self, raw_cls_bbox, raw_prob, opts):

        raw_cls_bbox = raw_cls_bbox.cpu()
        raw_prob = raw_prob.cpu()

        bbox = list()
        label = list()
        score = list()

        for l in range(1, self.num_classes):
            prob_l = raw_prob[..., l]
            mask = prob_l > opts.conf_thres
            cls_bbox_l = raw_cls_bbox[mask]
            prob_l = prob_l[mask]
            keep = torchvision_nms(cls_bbox_l, prob_l, iou_threshold=0.45)
            bbox.append(cls_bbox_l[keep].cpu().numpy())
            label.append((l - 1) * np.ones(len(keep)))
            score.append(prob_l[keep].cpu().numpy())

        bbox = np.concatenate(bbox, axis=0).astype(np.float32)
        label = np.concatenate(label, axis=0).astype(np.int32)
        score = np.concatenate(score, axis=0).astype(np.float32)

        n_objects = score.shape[0]
        top_k = opts.top_k
        if n_objects > opts.top_k:
            sort_ind = score.argsort(axis=0)[::-1]  # [::-1] means descending
            score = score[sort_ind][:top_k]  # (top_k)
            bbox = bbox[sort_ind][:top_k]  # (top_k, 4)
            label = label[sort_ind][:top_k]  # (top_k)
        return bbox, label, score


if __name__ == '__main__':
    img = torch.randn([1, 3, 416, 416])
    model = YoloV3(Darknet53(pretrained=True))
    p_l, p_m, p_s = model(img)

    print("large : ", p_l.size())
    print("medium : ", p_m.size())
    print("small : ", p_s.size())

    '''
    torch.Size([1, 52, 52, 255])
    torch.Size([1, 26, 26, 255])
    torch.Size([1, 13, 13, 255])
    '''

