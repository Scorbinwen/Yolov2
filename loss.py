import torch
import torch.nn as nn
from utils import GetIouBetween
import config


# 由于网上关于yolov2的Loss形式各异，因此这里研究Darknet19的源码forward.c
class YoloLoss(nn.Module):
    def __init__(self):
        self.fm_width = config.output_width
        self.fm_height = config.output_height
        self.anchor_num = config.anchor_num
        self.class_num = config.class_num
        self.batchsize = config.batch_size
        self.iou_threshold = config.iou_threshold
        self.mse_loss = torch.nn.MSELoss(reduction="mean")
        self.scale_noobj = config.scale_noobj
        self.scale_obj = config.scale_obj
        self.sigmoid = torch.nn.Sigmoid()

        x = torch.range(0, self.fm_width, requires_grad=False)
        y = torch.range(0, self.fm_height, requires_grad=False)
        x_cord, y_cord = torch.meshgrid(x, y)
        self.fm_cord = torch.concat((x_cord[..., None], y_cord[..., None]), dim=-1)

        self.anchor_box = torch.zeros(self.batchsize, self.fm_width, self.fm_height, self.anchor_num, 2,
                                      requires_grad=False)
        self.anchor_box[..., :, :] = torch.Tensor(config.anchor_box)

        self.obj_mask = torch.zeros(self.batchsize, self.fm_width, self.fm_height, self.anchor_num, 1,
                                    requires_grad=False)
        self.true_bbox = torch.zeros(self.batchsize, self.fm_width, self.fm_height, self.anchor_num, 4, requires_grad=False)
        self.true_score = torch.zeros(self.batchsize, self.fm_width, self.fm_height, self.anchor_num, self.class_num, requires_grad=False)
        self.iou = torch.zeros_like(self.obj_mask, requires_grad=False)

    def __call__(self, iter, pred, target):
        if iter is None:
            raise ValueError("iter can not be none!")
        need_prior_loss = iter < config.anchor_train_iters

        cls_score, pred_object = pred
        true_label, true_object = target
        # FIX ME: add raise Error for input check
        # expect cls_score : (N, W(13), H(13), 4, class_num)
        # expect pred_object: (N, W(13), H(13), 4, 5)
        # expect true_object: (N, obj_num, 4)
        # expect true_label : (N, obj_num, class_num)

        cls_score = torch.reshape(cls_score, [-1, self.fm_width, self.fm_height, self.anchor_num, self.class_num])
        pred_object = torch.reshape(pred_object, [-1, self.fm_width, self.fm_height, self.anchor_num, 5])
        # the 5 value of pred_object[...,:] is t_x, t_y, t_w, t_h, iou_pred
        # the 5 value of true_object[...,:] is centroid(x, y), w, h s.t x, y within (0, W) and w, h within (0, H)
        for anchor_ind in range(self.anchor_num):
            pred_object[..., anchor_ind, :2] = self.sigmoid(pred_object[:, :, :, anchor_ind, :2]) + self.fm_cord[:, :,
                                                                                                    :2]
            pred_object[..., anchor_ind, :2] = torch.exp(pred_object[:, :, :, anchor_ind, 2:4]) * self.anchor_box[:, :,
                                                                                                  anchor_ind, :2]

        true_object_center_x = torch.ceil(true_object[:, :, 0] / self.fm_width)
        true_object_center_y = torch.ceil(true_object[:, :, 1] / self.fm_height)
        true_object_center = torch.zeros_like(self.batchsize, self.fm_width, self.fm_height, self.anchor_num, 4)

        for b in range(self.batchsize):
            truebbox_index = 0
            for i, j in true_object_center_x, true_object_center_y:
                true_object_center[b, i, j, :, :] = true_object[b, truebbox_index, :]
                for anchor_ind in range(self.anchor_num):
                    _iou = GetIouBetween(true_object_center[b, i, j, anchor_ind, :], true_object[b, truebbox_index, :])
                    if _iou > self.iou_threshold:
                        self.obj_mask[b, i, j, anchor_ind, 0] = 1
                    self.iou[b, i, j, anchor_ind, 0] = _iou
                    self.true_bbox[b, i, j, anchor_ind, :] = true_object[b, truebbox_index, :]
                    self.true_score[b, i, j, anchor_ind, :] = true_label[b, truebbox_index, :]
                truebbox_index = truebbox_index + 1
        noobj_mask = 1 - self.obj_mask

        # objectness loss
        pred_object[..., 4:5] = self.sigmoid(pred_object[..., 4:5])
        noobj_loss = torch.mean(self.scale_noobj * noobj_mask * .5 * self.mse_loss(pred_object[..., 4:5], 0))
        obj_loss = torch.mean(
            self.scale_obj * self.obj_mask * .5 * self.mse_loss(pred_object[..., 4:5], self.iou[..., 0]))

        # bbox cordinate loss
        # anchor_box does not contribute to loss of x, y
        prior_loss = torch.mean(
            need_prior_loss * self.scale_obj * self.obj_mask * .5 * self.mse_loss(pred_object[..., 2:4],
                                                                                  self.anchor_box[..., :2]))
        true_loss = torch.mean(
            self.scale_obj * self.obj_mask * .5 * self.mse_loss(pred_object[..., :4], self.true_bbox[..., :4]))

        # class loss
        score_loss = torch.mean(self.scale_obj * self.obj_mask * .5 * self.mse_loss(cls_score, self.true_score))

        # the loss may drop suddenly at iter 12800
        total_loss = (noobj_loss + obj_loss + prior_loss + true_loss + score_loss) / 4.
        return total_loss
