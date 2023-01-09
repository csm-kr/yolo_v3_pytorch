import torch
import numpy as np
import torch.nn as nn
from utils.util import cxcy_to_xy, xy_to_cxcy, find_jaccard_overlap


class YoloV3Loss(nn.Module):
    def __init__(self, opts):
        super().__init__()

        self.num_classes = opts.num_classes
        self.mse = nn.MSELoss(reduction='none')
        self.bce = nn.BCELoss(reduction='none')
        self.resize = opts.resize

    def assign_target(self, b, n_obj, iou_anchors_gt, target_dict, scaled_center_gt_box, anchor_whs, label):

        size = target_dict['gt_target_xy'].shape[1]
        ratio_size = int(self.resize/size)

        cx, cy = scaled_center_gt_box[..., :2][n_obj]

        cx = int(cx)
        cy = int(cy)

        max_iou, max_idx = iou_anchors_gt[cy, cx, :, n_obj].max(0)  # which anchor has maximum iou? # print("max_iou : ", max_iou)
        j = max_idx  # j is idx.

        # # j-th anchor
        target_dict['gt_target_obj'][b, cy, cx, j, 0] = 1
        ratio_of_xy = scaled_center_gt_box[..., :2][n_obj] - scaled_center_gt_box[..., :2][n_obj].floor()
        target_dict['gt_target_xy'][b, cy, cx, j, :] = ratio_of_xy
        ratio_of_wh = scaled_center_gt_box[..., 2:][n_obj] / torch.from_numpy(np.array(anchor_whs[j]) / ratio_size).to(torch.get_device(scaled_center_gt_box))
        target_dict['gt_target_wh'][b, cy, cx, j, :] = torch.log(ratio_of_wh)
        target_dict['gt_target_cls'][b, cy, cx, j, int(label[n_obj] + 1)] = 1

        return target_dict

    def make_scaled_iou(self, size, corner_anchor, scaled_center_gt_box):
        scaled_corner_gt_box = cxcy_to_xy(scaled_center_gt_box)
        iou_anchors_gt = find_jaccard_overlap(corner_anchor, scaled_corner_gt_box)  # [13 * 13 * 3, # obj]
        iou_anchors_gt = iou_anchors_gt.view(size, size, 3, -1)
        return iou_anchors_gt

    def make_scaled_center_gt_box(self, size, corner_gt_box):
        center_gt_box = xy_to_cxcy(corner_gt_box)                                   # (0 ~ 1) cx cy w h
        scaled_center_gt_box = center_gt_box * float(size)                          # (0 ~ out_size) x1 y1 x2 y2
        return scaled_center_gt_box

    def make_target_container(self, batch_size, grid_size, device):
        gs = grid_size
        target_dict = {}
        target_dict['ignore_mask'] = torch.zeros([batch_size, gs, gs, 3]).to(device)
        target_dict['gt_target_xy'] = torch.zeros([batch_size, gs, gs, 3, 2]).to(device)
        target_dict['gt_target_wh'] = torch.zeros([batch_size, gs, gs, 3, 2]).to(device)
        target_dict['gt_target_obj'] = torch.zeros([batch_size, gs, gs, 3, 1]).to(device)
        target_dict['gt_target_cls'] = torch.zeros([batch_size, gs, gs, 3, self.num_classes]).to(device)
        return target_dict

    def make_target(self, gt_boxes, gt_labels, anchor):
        '''
        gt_boxes : list of tensor x1y1x2y2 shape boxes range : 0 ~ 1, shape : []
        gt_labels :
        anchor : YoloV3Anchor class
        '''

        device = gt_boxes[0].get_device()

        center_anchor_l, center_anchor_m, center_anchor_s = anchor.center_anchors
        corner_anchor_l = cxcy_to_xy(center_anchor_l).view(13 * 13 * 3, 4)
        corner_anchor_m = cxcy_to_xy(center_anchor_m).view(26 * 26 * 3, 4)
        corner_anchor_s = cxcy_to_xy(center_anchor_s).view(52 * 52 * 3, 4)

        # 1. make target container
        batch_size = len(gt_labels)
        target_dict_l = self.make_target_container(batch_size, grid_size=center_anchor_l.shape[0], device=device)
        target_dict_m = self.make_target_container(batch_size, grid_size=center_anchor_m.shape[0], device=device)
        target_dict_s = self.make_target_container(batch_size, grid_size=center_anchor_s.shape[0], device=device)

        for b in range(batch_size):

            label = gt_labels[b]
            corner_gt_box = gt_boxes[b]                             # (0 ~ 1) x1 y1 x2 y2
            size = {"l": 13, "m": 26, "s": 52}

            # 2. make scaled center gt box for target
            scaled_center_gt_box_l = self.make_scaled_center_gt_box(size=size['l'], corner_gt_box=corner_gt_box)
            scaled_center_gt_box_m = self.make_scaled_center_gt_box(size=size['m'], corner_gt_box=corner_gt_box)
            scaled_center_gt_box_s = self.make_scaled_center_gt_box(size=size['s'], corner_gt_box=corner_gt_box)

            # 3. conduct iou between (corner) anchor and (corner) gt
            iou_anchors_gt_l = self.make_scaled_iou(size['l'], corner_anchor_l, scaled_center_gt_box_l)
            iou_anchors_gt_m = self.make_scaled_iou(size['m'], corner_anchor_m, scaled_center_gt_box_m)
            iou_anchors_gt_s = self.make_scaled_iou(size['s'], corner_anchor_s, scaled_center_gt_box_s)

            # *********************** assign target ***********************
            num_obj = corner_gt_box.size(0)
            for n_obj in range(num_obj):

                # 4. find best anchor
                best_idx = torch.FloatTensor(
                    [iou_anchors_gt_l[..., n_obj].max(),
                     iou_anchors_gt_m[..., n_obj].max(),
                     iou_anchors_gt_s[..., n_obj].max()]).argmax()

                if best_idx == 0:
                    anchor_whs_l = anchor.anchor_whs['large']
                    target_dict_l = self.assign_target(b, n_obj, iou_anchors_gt_l, target_dict_l,
                                                       scaled_center_gt_box_l, anchor_whs_l, label)
                if best_idx == 1:
                    anchor_whs_m = anchor.anchor_whs['middle']
                    target_dict_m = self.assign_target(b, n_obj, iou_anchors_gt_m, target_dict_m,
                                                       scaled_center_gt_box_m, anchor_whs_m, label)
                if best_idx == 2:
                    anchor_whs_s = anchor.anchor_whs['small']
                    target_dict_s = self.assign_target(b, n_obj, iou_anchors_gt_s, target_dict_s,
                                                       scaled_center_gt_box_s, anchor_whs_s, label)

            # ignore_mask
            target_dict_l['ignore_mask'][b] = (iou_anchors_gt_l.max(-1)[0] < 0.5)
            target_dict_m['ignore_mask'][b] = (iou_anchors_gt_m.max(-1)[0] < 0.5)
            target_dict_s['ignore_mask'][b] = (iou_anchors_gt_s.max(-1)[0] < 0.5)

        return [target_dict_l, target_dict_m, target_dict_s]

    def yolo_v3_loss(self, pred_targets, gt_target_dict):

        pred_target_xy, pred_target_wh, pred_obj, pred_cls = pred_targets

        # xy_loss = gt_target_dict['gt_target_obj'] * self.mse(pred_target_xy, gt_target_dict['gt_target_xy'])
        # wh_loss = gt_target_dict['gt_target_obj'] * self.mse(pred_target_wh, gt_target_dict['gt_target_wh'])
        # obj_loss = gt_target_dict['gt_target_obj'] * self.bce(pred_obj, gt_target_dict['gt_target_obj'])
        # no_obj_loss = (1 - gt_target_dict['gt_target_obj']) * self.bce(pred_obj, gt_target_dict['gt_target_obj']) * gt_target_dict['ignore_mask'].unsqueeze(-1)
        # cls_loss = gt_target_dict['gt_target_obj'] * self.bce(pred_cls, gt_target_dict['gt_target_cls'])

        xy_loss = torch.mean(self.mse(pred_target_xy, gt_target_dict['gt_target_xy']), dim=-1) * gt_target_dict['gt_target_obj'].squeeze(-1)
        wh_loss = torch.mean(self.mse(pred_target_wh, gt_target_dict['gt_target_wh']), dim=-1) * gt_target_dict['gt_target_obj'].squeeze(-1)
        obj_loss = gt_target_dict['gt_target_obj'] * self.bce(pred_obj, gt_target_dict['gt_target_obj'])
        no_obj_loss = (1 - gt_target_dict['gt_target_obj']) * self.bce(pred_obj, gt_target_dict['gt_target_obj']) * gt_target_dict['ignore_mask'].unsqueeze(-1)
        cls_loss = gt_target_dict['gt_target_obj'] * self.bce(pred_cls, gt_target_dict['gt_target_cls'])
        return xy_loss, wh_loss, obj_loss, no_obj_loss, cls_loss

    def pred2target(self, pred):
        # pred to target
        out_size = pred.size(1)  # 13, 13
        pred_targets = pred.view(-1, out_size, out_size, 3, 5 + self.num_classes)
        pred_target_xy = pred_targets[..., :2].sigmoid()                # 0, 1 sigmoid(tx, ty) -> bx, by
        pred_target_wh = pred_targets[..., 2:4]                         # 2, 3
        pred_objectness = pred_targets[..., 4].unsqueeze(-1).sigmoid()  # 4        class probability
        pred_classes = pred_targets[..., 5:].sigmoid()                  # 21 / 81  classes
        return pred_target_xy, pred_target_wh, pred_objectness, pred_classes

    def forward(self, pred, gt_boxes, gt_labels, anchor):
        """
        :param pred_targets_1: (B, 13, 13, 255)
        :param pred_targets_2: (B, 26, 26, 255)
        :param pred_targets_3: (B, 52, 52, 255)

        :param pred_xy_1 : (B, 13, 13, 2)
        :param pred_wh_1 : (B, 13, 13, 2)

        :param gt_boxes:     (B, 4)
        :param gt_labels:    (B)
        :return:
        """
        batch_size = len(gt_labels)
        pred_l, pred_m, pred_s = pred

        # make target dictionary
        target_l, target_m, target_s = self.make_target(gt_boxes, gt_labels, anchor)

        # get loss pred_target and gt_target dict
        xy_loss_l, wh_loss_l, obj_loss_l, no_obj_loss_l, cls_loss_l = self.yolo_v3_loss(self.pred2target(pred_l), target_l)
        xy_loss_m, wh_loss_m, obj_loss_m, no_obj_loss_m, cls_loss_m = self.yolo_v3_loss(self.pred2target(pred_m), target_m)
        xy_loss_s, wh_loss_s, obj_loss_s, no_obj_loss_s, cls_loss_s = self.yolo_v3_loss(self.pred2target(pred_s), target_s)

        loss1 = 5 * (xy_loss_l.sum() + xy_loss_m.sum() + xy_loss_s.sum()) / batch_size
        loss2 = 5 * (wh_loss_l.sum() + wh_loss_m.sum() + wh_loss_s.sum()) / batch_size
        loss3 = 1 * (obj_loss_l.sum() + obj_loss_m.sum() + obj_loss_s.sum()) / batch_size
        loss4 = 0.5 * (no_obj_loss_l.sum() + no_obj_loss_m.sum() + no_obj_loss_s.sum()) / batch_size
        loss5 = 1 * (cls_loss_l.sum() + cls_loss_m.sum() + cls_loss_s.sum()) / batch_size

        return loss1 + loss2 + loss3 + loss4 + loss5, (loss1, loss2, loss3, loss4, loss5), (target_l, target_m, target_s)
