import torch
import torch.nn as nn
from utils import GetCenterAlignIouBetween
import config
import torch.nn.functional as F


class YoloLoss(nn.Module):
    def __init__(self):
        super(YoloLoss, self).__init__()
        self.fm_width = config.output_size
        self.fm_height = config.output_size
        self.anchor_num = config.anchor_num
        self.class_num = config.class_num
        self.batchsize = config.batch_size
        self.iou_threshold = config.iou_threshold
        self.scale_noobj = config.scale_noobj
        self.scale_obj = config.scale_obj
        self.downsample_rate = config.downsample_rate
        self.fm_cord = config.fm_cord

        self.conf_loss_function = nn.BCELoss(reduction='none')
        self.cls_loss_function = nn.CrossEntropyLoss(reduction='none')
        self.txty_loss_function = nn.MSELoss(reduction='none')
        self.twth_loss_function = nn.MSELoss(reduction='none')


    def __call__(self, epoch, pred, target):

        conf, pred_xy, pred_wh, cls_score, cls_out = pred
        self.obj_mask = torch.zeros(self.batchsize, self.fm_width, self.fm_height, self.anchor_num, 1,
                                    requires_grad=False)
        self.true_bbox = torch.zeros(self.batchsize, self.fm_width, self.fm_height, self.anchor_num, 4,
                                     requires_grad=False)
        self.true_score = torch.zeros(self.batchsize, self.fm_width, self.fm_height, self.anchor_num, self.class_num,
                                      requires_grad=False)
        self.true_score[..., -1] = 1
        self.pred_gt_iou = torch.zeros_like(self.obj_mask, requires_grad=False)
        self.scale_weight = torch.zeros_like(self.obj_mask, requires_grad=False)
        self.anchor_gt_iou = torch.zeros(self.anchor_num)

        if config.train_detection:
            # pred h, w
            for b in range(self.batchsize):
                _, true_label, true_object = target[b]
                true_object = true_object / self.downsample_rate
                truebbox_index = -1
                # 遍历所有 (N, W, H, anchor_num)个anchor，比较其与true_object 之间的iou，
                # 实际上只有true_object和anchor的中心在同一个bin的时候，anchor才有可能成为正样本，其余anchor的标签一定不可能是该true_object.
                for i, j in zip(true_object[..., 1].tolist(), true_object[..., 0].tolist()):
                    truebbox_index = truebbox_index + 1
                    i = int(i)
                    j = int(j)
                    # 从anchor_num个anchor中选取最大的iou对应的predbox作为正样本
                    gt_bbox = torch.tensor([0, 0, true_object[truebbox_index, 2], true_object[truebbox_index, 3]])
                    # print("gt_bbox", gt_bbox)
                    # 后续将该for循环改成矩阵运算，加快训练以及减小代码复杂度
                    for anchor_ind in range(self.anchor_num):
                        # anchor box的x, y 均为中心坐标。
                        anchor_bbox = torch.tensor(
                            [0, 0, config.anchor_box[anchor_ind][0], config.anchor_box[anchor_ind][1]])
                        # print("anchor_bbox", anchor_bbox)
                        self.anchor_gt_iou[anchor_ind] = GetCenterAlignIouBetween(anchor_bbox, gt_bbox)

                    best_iou_index = torch.argmax(self.anchor_gt_iou)
                    print("i:{} j:{} best_iou_index:{} best_iou:{}".format(i, j, best_iou_index, self.anchor_gt_iou[best_iou_index]))
                    # 将best_iou_index 对应的target设置为正样本。
                    self.true_bbox[b, i, j, best_iou_index, :] = true_object[truebbox_index, :]
                    # print("cls_score", cls_score[b, i, j, best_iou_index,:])
                    self.obj_mask[b, i, j, best_iou_index] = 1
                    # print("true_label", true_label.shape)
                    self.true_score[b, i, j, best_iou_index, :] = true_label[truebbox_index, :]
                    # print("pred_xy[b, i, j, best_iou_index]", pred_xy[b, i, j, best_iou_index])
                    self.pred_gt_iou[b, i, j, best_iou_index] = GetCenterAlignIouBetween(
                        torch.cat((pred_xy[b, i, j, best_iou_index], pred_wh[b, i, j, best_iou_index])),
                        true_object[truebbox_index, :])
                    # print("self.pred_gt_iou[b, i, j, best_iou_index]", self.pred_gt_iou[b, i, j, best_iou_index])
                    # print("conf[b, i, j, best_iou_index]", conf[b, i, j, best_iou_index])
                    self.scale_weight[b, i, j, best_iou_index] = \
                        2 - (true_object[truebbox_index, 2] / config.output_size) * \
                        (true_object[truebbox_index, 3] / config.output_size)

                    for anchor_ind in range(self.anchor_num):
                        if anchor_ind != best_iou_index:
                            if self.anchor_gt_iou[anchor_ind] > self.iou_threshold:
                                # 这些超过iou阈值的anchor box不加入训练，主要目的是为了将anchor bbox分化成不同的anchor size。
                                self.obj_mask[b, i, j, anchor_ind] = -1
                                self.scale_weight[b, i, j, anchor_ind] = -1
                            else:
                                # 取iou最大的，其余为负样本。
                                self.obj_mask[b, i, j, anchor_ind] = 0
                                self.pred_gt_iou[b, i, j, anchor_ind] = GetCenterAlignIouBetween(
                                    torch.cat((pred_xy[b, i, j, best_iou_index], pred_wh[b, i, j, best_iou_index])),
                                    true_object[truebbox_index, :])

            tmp_pred_wh = pred_wh.view(self.batchsize, self.fm_width*self.fm_height*self.anchor_num, 2).contiguous()
            tmp_pred_xy = pred_xy.view(self.batchsize, self.fm_width*self.fm_height*self.anchor_num, 2).contiguous()
            tmp_true_bbox = self.true_bbox.view(self.batchsize, self.fm_width*self.fm_height*self.anchor_num, 4).contiguous()
            tmp_obj_mask = self.obj_mask.view(self.batchsize, self.fm_width*self.fm_height*self.anchor_num, 1).contiguous()
            tmp_true_score = self.true_score.view(self.batchsize, self.fm_width*self.fm_height*self.anchor_num, self.class_num).permute(0, 2, 1).contiguous()
            tmp_cls_score = cls_score.view(self.batchsize, self.fm_width*self.fm_height*self.anchor_num, self.class_num).permute(0, 2, 1).contiguous()
            tmp_pred_gt_iou = self.pred_gt_iou.view(self.batchsize, self.fm_width*self.fm_height*self.anchor_num, 1).contiguous()
            tmp_scale_weight = self.scale_weight.view(self.batchsize, self.fm_width*self.fm_height*self.anchor_num, 1).contiguous()
            tmp_conf = conf.view(self.batchsize, self.fm_width*self.fm_height*self.anchor_num, 1).contiguous()
            tmp_fm_cord = config.fm_cord.view(self.fm_width*self.fm_height, 2).repeat(1, self.anchor_num, 1).contiguous()
            # objectness loss
            print("tmp_conf max", torch.max(tmp_conf))
            index_conf = torch.argmax(tmp_conf.squeeze())
            print("index conf", index_conf)
            index_mask = torch.argmax(tmp_obj_mask.squeeze())
            print("index mask", index_mask)
            # print("xy{} wh:{}".format(tmp_pred_xy[index], tmp_pred_wh[index]))
            noobj_loss = (
                self.scale_noobj * (tmp_obj_mask == 0) * self.conf_loss_function(tmp_conf, torch.zeros_like(tmp_conf))).sum() / self.batchsize
            obj_loss = (
                self.scale_obj * (tmp_obj_mask == 1) * self.conf_loss_function(tmp_conf, tmp_pred_gt_iou)).sum() / self.batchsize

            gt_mask = (tmp_scale_weight > 0)
            true_loss_xy = (self.scale_obj * gt_mask * tmp_scale_weight *
                                      self.txty_loss_function(
                                          tmp_pred_xy - tmp_fm_cord[None, ...],
                                          tmp_true_bbox[..., :2] - tmp_fm_cord[None, ...])).sum() / self.batchsize

            true_loss_wh = (self.scale_obj * gt_mask * tmp_scale_weight *
                                      self.twth_loss_function(
                                          tmp_pred_wh, tmp_true_bbox[..., 2:])).sum() / self.batchsize

            cls_gt_mask = gt_mask.permute(0, 2, 1).contiguous()
            index = torch.where(cls_gt_mask == 1)
            print("index", index)
            _cls_gt_mask = gt_mask.permute(0, 2, 1).contiguous().squeeze(dim=1)
            index = torch.where(_cls_gt_mask == 1)
            print("tmp_cls_score shape", tmp_cls_score.shape)
            print("tmp_cls_score", tmp_cls_score.permute(0, 2, 1)[index])
            print("tmp_true_score", tmp_true_score.permute(0, 2, 1)[index])
            print("pred result", torch.argmax(tmp_cls_score.permute(0, 2, 1)[index], dim=-1) == torch.argmax(tmp_true_score.permute(0, 2, 1)[index], dim=-1))

            score_loss = (self.scale_obj * cls_gt_mask * self.cls_loss_function(tmp_cls_score, tmp_true_score)).sum() / self.batchsize

            total_loss = noobj_loss + obj_loss + true_loss_xy + true_loss_wh + score_loss
            total_loss = score_loss
            print("cls_loss:{}".format(score_loss))
        else:
            imagenet_label = []
            for b in range(self.batchsize):
                _, true_label, _ = target[b]
                imagenet_label.append(true_label)
            imagenet_label = torch.cat(imagenet_label, dim=0)
            print("imagenet_label", imagenet_label)
            print("cls_out", cls_out)

            imagenet_label = torch.as_tensor(imagenet_label, dtype=torch.float)
            print("image_label", imagenet_label.shape)
            imagenet_loss = torch.mean(self.cls_loss_function(cls_out, imagenet_label))
            print(imagenet_loss)
            total_loss = imagenet_loss
            noobj_loss = torch.tensor(0.)
            obj_loss = torch.tensor(0.)
            score_loss = torch.tensor(0.)
            true_loss_wh = torch.tensor(0.)
            true_loss_xy = torch.tensor(0.)

        return total_loss, noobj_loss / 4., obj_loss / 4., score_loss / 4., true_loss_xy / 4., true_loss_wh / 4.
