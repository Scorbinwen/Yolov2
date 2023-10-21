import torch.nn as nn
from collections import OrderedDict
import config
import torch
import numpy as np
import torch.nn.functional as F


class Conv(nn.Module):
    def __init__(self, in_ch, out_ch, k=1, p=0, s=1, d=1, act=True):
        super(Conv, self).__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, dilation=d, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.1, inplace=True) if act else nn.Identity()
        )

    def forward(self, x):
        return self.convs(x)


class Conv_BN_LeakyReLU(nn.Module):
    def __init__(self, in_channels, out_channels, ksize, padding=0, stride=1, dilation=1):
        super(Conv_BN_LeakyReLU, self).__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, ksize, padding=padding, stride=stride, dilation=dilation),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1, inplace=True)
        )

    def forward(self, x):
        return self.convs(x)


class DarkNet_19(nn.Module):
    def __init__(self):
        super(DarkNet_19, self).__init__()
        # backbone network : DarkNet-19
        # output : stride = 2, c = 32
        self.conv_1 = nn.Sequential(
            Conv_BN_LeakyReLU(3, 32, 3, 1),
            nn.MaxPool2d((2, 2), 2),
        )

        # output : stride = 4, c = 64
        self.conv_2 = nn.Sequential(
            Conv_BN_LeakyReLU(32, 64, 3, 1),
            nn.MaxPool2d((2, 2), 2)
        )

        # output : stride = 8, c = 128
        self.conv_3 = nn.Sequential(
            Conv_BN_LeakyReLU(64, 128, 3, 1),
            Conv_BN_LeakyReLU(128, 64, 1),
            Conv_BN_LeakyReLU(64, 128, 3, 1),
            nn.MaxPool2d((2, 2), 2)
        )

        # output : stride = 8, c = 256
        self.conv_4 = nn.Sequential(
            Conv_BN_LeakyReLU(128, 256, 3, 1),
            Conv_BN_LeakyReLU(256, 128, 1),
            Conv_BN_LeakyReLU(128, 256, 3, 1),
        )

        # output : stride = 16, c = 512
        self.maxpool_4 = nn.MaxPool2d((2, 2), 2)
        self.conv_5 = nn.Sequential(
            Conv_BN_LeakyReLU(256, 512, 3, 1),
            Conv_BN_LeakyReLU(512, 256, 1),
            Conv_BN_LeakyReLU(256, 512, 3, 1),
            Conv_BN_LeakyReLU(512, 256, 1),
            Conv_BN_LeakyReLU(256, 512, 3, 1),
        )

        # output : stride = 32, c = 1024
        self.maxpool_5 = nn.MaxPool2d((2, 2), 2)
        self.conv_6 = nn.Sequential(
            Conv_BN_LeakyReLU(512, 1024, 3, 1),
            Conv_BN_LeakyReLU(1024, 512, 1),
            Conv_BN_LeakyReLU(512, 1024, 3, 1),
            Conv_BN_LeakyReLU(1024, 512, 1),
            Conv_BN_LeakyReLU(512, 1024, 3, 1),
        )
        self.first = True

    def forward(self, x):
        """
        Input:
            x: (Tensor) -> [B, 3, H, W]
        Output:
            output: (Dict) {
                'c3': c3 -> Tensor[B, C3, H/8, W/8],
                'c4': c4 -> Tensor[B, C4, H/16, W/16],
                'c5': c5 -> Tensor[B, C5, H/32, W/32],
            }
        """
        c1 = self.conv_1(x)  # [B, C1, H/2, W/2]
        c2 = self.conv_2(c1)  # [B, C2, H/4, W/4]
        c3 = self.conv_3(c2)  # [B, C3, H/8, W/8]
        c3 = self.conv_4(c3)  # [B, C3, H/8, W/8]
        c4 = self.conv_5(self.maxpool_4(c3))  # [B, C4, H/16, W/16]
        c5 = self.conv_6(self.maxpool_5(c4))  # [B, C5, H/32, W/32]

        output = {
            'c3': c3,
            'c4': c4,
            'c5': c5
        }
        # if (self.first == True):
        #     # print("running mean shape:{} val:{}".format(self.conv_1[0].convs[1].running_mean.shape,
        #     #                                             self.conv_1[0].convs[1].running_mean))
        #     # print("running var  shape:{} val:{}".format(self.conv_1[0].convs[1].running_var.shape,
        #     #                                             self.conv_1[0].convs[1].running_var))
        #     np.savetxt('c5.csv',
        #                c5.view(-1, 1024 * 13 * 13).detach().cpu().numpy(), fmt='%.2f', delimiter=',')
        #     self.first = False
        return output


class reorg_layer(nn.Module):
    def __init__(self, stride):
        super(reorg_layer, self).__init__()
        self.stride = stride

    def forward(self, x):
        batch_size, channels, height, width = x.size()
        _height, _width = height // self.stride, width // self.stride

        _x = x.view(batch_size, channels, _height, self.stride, _width, self.stride).transpose(3, 4).contiguous()
        _x = _x.view(batch_size, channels, _height * _width, self.stride * self.stride).transpose(2, 3).contiguous()
        _x = _x.view(batch_size, channels, self.stride * self.stride, _height, _width).transpose(1, 2).contiguous()
        _x = _x.view(batch_size, -1, _height, _width)

        return _x


def build_darknet19(pretrained=False):
    # model
    model = DarkNet_19()
    feat_dims = [256, 512, 1024]

    # load weight
    if pretrained:
        print('Loading pretrained weight ...')
        # checkpoint state dict
        checkpoint_state_dict = torch.load(config.backbone_state_dict)
        # model state dict
        model_state_dict = model.state_dict()
        # check
        for k in list(checkpoint_state_dict.keys()):
            if k in model_state_dict:
                shape_model = tuple(model_state_dict[k].shape)
                shape_checkpoint = tuple(checkpoint_state_dict[k].shape)
                if shape_model != shape_checkpoint:
                    checkpoint_state_dict.pop(k)
            else:
                checkpoint_state_dict.pop(k)
                print(k)

        model.load_state_dict(checkpoint_state_dict)

    return model, feat_dims


class Yolov2(nn.Module):
    def __init__(self, trainable=False):
        super(Yolov2, self).__init__()
        self.trainable = trainable
        self.input_size = config.input_size
        self.num_classes = config.class_num
        self.num_anchors = config.anchor_num
        # 主干网络
        self.backbone, feat_dims = build_darknet19(config.pretrained_backbone)

        # 检测头
        self.convsets_1 = nn.Sequential(
            Conv(feat_dims[-1], config.head_dim, k=3, p=1),
            Conv(config.head_dim, config.head_dim, k=3, p=1)
        )
        self.sigmoid = torch.nn.Sigmoid()
        # 融合高分辨率的特征信息
        self.route_layer = Conv(feat_dims[-2], config.reorg_dim, k=1)
        self.reorg = reorg_layer(stride=2)

        # 检测头
        self.convsets_2 = Conv(config.head_dim + config.reorg_dim * 4, config.head_dim, k=3, p=1)

        # 预测层
        self.pred = nn.Conv2d(config.head_dim, self.num_anchors * (1 + 4 + self.num_classes), 1)

        # self.classifier = nn.Sequential(
        #     nn.Conv2d(config.head_dim, config.class_num, kernel_size=1),
        #     nn.AdaptiveAvgPool2d((1, 1)),
        # )
        self.softmax = nn.Softmax(dim=-1)
        self.first = True

        if self.trainable:
            self.init_bias()

    def init_bias(self):
        # init bias
        init_prob = 0.01
        bias_value = -torch.log(torch.tensor((1. - init_prob) / init_prob))
        nn.init.constant_(self.pred.bias[..., :self.num_anchors], bias_value)
        nn.init.constant_(self.pred.bias[..., 1 * self.num_anchors:(1 + self.num_classes) * self.num_anchors],
                          bias_value)

    def forward(self, x, targets=None):
        # backbone主干网络
        feats = self.backbone(x)
        c4, c5 = feats['c4'], feats['c5']
        # 处理c5特征
        p5 = self.convsets_1(c5)
        # 融合c4特征
        p4 = self.reorg(self.route_layer(c4))
        p5 = torch.cat([p4, p5], dim=1)

        # 处理p5特征
        p5 = self.convsets_2(p5)

        # 预测
        prediction = self.pred(p5)
        prediction = prediction.permute(0, 2, 3, 1).contiguous().view(-1, config.output_size, config.output_size,
                                     self.num_anchors * (1 + 4 + self.num_classes))

        conf = prediction[..., :self.num_anchors]
        conf = conf.contiguous().view(-1, config.output_size, config.output_size, self.num_anchors, 1)
        conf = self.sigmoid(conf)

        cls_score = prediction[..., self.num_anchors:(self.num_classes + 1) * self.num_anchors]
        cls_score = cls_score.contiguous().view(-1, config.output_size, config.output_size, self.num_anchors,
                                                self.num_classes)

        pred_xy = prediction[..., (self.num_classes + 1) * self.num_anchors:
                                  (self.num_classes + 1) * self.num_anchors + 2 * self.num_anchors]
        pred_xy = pred_xy.contiguous().view(-1, config.output_size, config.output_size, self.num_anchors, 2)
        pred_xy = self.sigmoid(pred_xy) + config.fm_cord[..., None, :2]

        pred_wh = prediction[..., (self.num_classes + 1) * self.num_anchors + 2 * self.num_anchors:]
        pred_wh = pred_wh.contiguous().view(-1, config.output_size, config.output_size, self.num_anchors, 2)
        pred_wh = torch.exp(pred_wh) * torch.tensor(config.anchor_box)
        if (self.first == True):
            # for name, param in self.pred.named_parameters():
            #     print("name:{} param:{}".format(name, param))
            # print("running mean shape:{} val:{}".format(self.conv_1[0].convs[1].running_mean.shape,
            #                                             self.conv_1[0].convs[1].running_mean))
            # print("running var  shape:{} val:{}".format(self.conv_1[0].convs[1].running_var.shape,
            #                                             self.conv_1[0].convs[1].running_var))
            np.savetxt('prediction.csv',
                       prediction.view(-1, config.output_size * config.output_size *
                                       self.num_anchors * (1 + 4 + self.num_classes)).detach().cpu().numpy(),
                                        fmt='%.8f', delimiter=',')
            np.savetxt('c5.csv',
                       c5.view(-1, 13 * 13 * 1024).detach().cpu().numpy(),
                       fmt='%.8f', delimiter=',')
            np.savetxt('c4.csv',
                       c4.view(-1, 26 * 26 * 512).detach().cpu().numpy(),
                       fmt='%.8f', delimiter=',')
            np.savetxt('p5.csv',
                       p5.view(-1, 13 * 13 * 1024).detach().cpu().numpy(),
                       fmt='%.15f', delimiter=',')
            np.savetxt('cls_pred.csv',
                       cls_score.view(-1, config.output_size * config.output_size * self.num_anchors *
                                                self.num_classes).detach().cpu().numpy(),
                       fmt='%.8f', delimiter=',')
            self.first = False
        cls_out = None
        return conf, pred_xy, pred_wh, cls_score, cls_out

