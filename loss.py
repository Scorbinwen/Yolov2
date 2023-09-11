import torch
import torch.nn as nn
from utils import GetCenterAlignIouBetween
import config


class YoloLoss(nn.Module):
    def __init__(self):
        super(YoloLoss, self).__init__()
        self.fm_width = config.output_width
        self.fm_height = config.output_height
        self.anchor_num = config.anchor_num
        self.class_num = config.class_num
        self.batchsize = config.batch_size
        self.iou_threshold = config.iou_threshold
        self.mse_loss = torch.nn.MSELoss(reduce=False)
        self.scale_noobj = config.scale_noobj
        self.scale_obj = config.scale_obj
        self.sigmoid = torch.nn.Sigmoid()
        self.downsample_rate = config.downsample_rate

        self.fm_cord = config.fm_cord
        self.fm_size_limit = config.fm_size_limit
        self.anchor_box = torch.zeros(self.batchsize, self.fm_width, self.fm_height, self.anchor_num, 2,
                                      requires_grad=False)
        self.anchor_box[..., :, :] = torch.Tensor(config.anchor_box) * self.fm_width

        self.obj_mask = torch.zeros(self.batchsize, self.fm_width, self.fm_height, self.anchor_num, requires_grad=False)
        self.true_bbox = torch.zeros(self.batchsize, self.fm_width, self.fm_height, self.anchor_num, 4,
                                     requires_grad=False)
        self.true_score = torch.zeros(self.batchsize, self.fm_width, self.fm_height, self.anchor_num, self.class_num,
                                      requires_grad=False)
        self.true_score[..., -1] = 1
        self.iou = torch.zeros_like(self.obj_mask, requires_grad=False)

    def __call__(self, epoch, pred, target):
        need_prior_loss = epoch < config.anchor_train_epochs

        cls_score, pred_object = pred
        # expect cls_score : (N, W(13), H(13), 4, class_num)
        # expect pred_object: (N, W(13), H(13), 4, 5)
        # expect true_object: (N, obj_num, 4)
        # expect true_label : (N, obj_num, class_num)

        cls_score = torch.reshape(cls_score, [-1, self.fm_width, self.fm_height, self.anchor_num, self.class_num])
        pred_object = torch.reshape(pred_object, [-1, self.fm_width, self.fm_height, self.anchor_num, 5])
        # the 5 value of pred_object[...,:] is t_x, t_y, t_w, t_h, iou_pred
        # the 5 value of true_object[...,:] is centroid(x, y), w, h s.t x, y within (0, W) and w, h within (0, H)
        # pred center x, y
        pred_object[..., :2] = self.sigmoid(pred_object[..., :2]) + self.fm_cord[..., None, :2]
        pred_object[..., 2:4] = self.sigmoid(pred_object[..., 2:4]) * self.fm_size_limit[..., None, :2]
        # pred h, w

        pred_object[..., 4] = self.sigmoid(pred_object[..., 4])

        for b in range(self.batchsize):
            _, true_label, true_object = target[b]
            true_object = true_object / self.downsample_rate
            truebbox_index = 0
            for i, j in zip(true_object[..., 0].tolist(), true_object[..., 1].tolist()):
                i = int(i)
                j = int(j)
                for anchor_ind in range(self.anchor_num):
                    _iou = GetCenterAlignIouBetween(pred_object[b, i, j, anchor_ind, :4],
                                                    true_object[truebbox_index, :])
                    # print("_iou", _iou)
                    if _iou > self.iou_threshold:
                        self.obj_mask[b, i, j, anchor_ind] = 1
                        self.iou[b, i, j, anchor_ind] = _iou
                        self.true_bbox[b, i, j, anchor_ind, :] = true_object[truebbox_index, :]
                        self.true_score[b, i, j, anchor_ind, :] = true_label[truebbox_index, :]
                truebbox_index = truebbox_index + 1
        noobj_mask = 1 - self.obj_mask

        # objectness loss
        noobj_loss = torch.mean(self.scale_noobj * noobj_mask * .5 * self.mse_loss(pred_object[..., 4],
                                                                                   torch.zeros_like(
                                                                                       pred_object[..., 4])))
        obj_loss = torch.mean(
            self.scale_obj * self.obj_mask * .5 * self.mse_loss(pred_object[..., 4].squeeze(axis=-1), self.iou))

        # bbox cordinate loss
        # anchor_box does not contribute to loss of x, y
        prior_loss = torch.mean(
            need_prior_loss * self.scale_obj * .5 * self.obj_mask[..., None] * self.mse_loss(pred_object[..., 2:4],
                                                                                             self.anchor_box[..., :2]))
        true_loss = torch.mean(self.scale_obj * .5 * self.obj_mask[..., None] * self.mse_loss(pred_object[..., :4],
                                                                                              self.true_bbox[..., :4]))

        # class loss with no obj_mask, so that the network can learn no object.
        score_loss = torch.mean(self.scale_obj * .5 * self.mse_loss(cls_score, self.true_score))
        # the loss may drop suddenly at iter 12800
        total_loss = (noobj_loss + obj_loss + prior_loss + true_loss + score_loss) / 4.
        return total_loss
