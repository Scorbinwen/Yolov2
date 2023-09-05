import torch.nn as nn
from collections import OrderedDict
import config


class Darknet19(nn.Module):
    def __init__(self, class_num=config.class_num, anchor_num=config.anchor_num):
        super(Darknet19, self).__init__()
        self.class_num = class_num
        self.anchor_num = anchor_num
        # FIX ME: Add BN and shortCut for model
        network_list = [
            ('conv1', nn.Conv2d(3, 32, kernel_size=3, stride=1)),
            ('pool1', nn.MaxPool2d(2, stride=2)),
            ('conv2', nn.Conv2d(32, 64, kernel_size=3, stride=1)),
            ('pool2', nn.MaxPool2d(2, stride=2)),
            ('conv3_1', nn.Conv2d(64, 128, kernel_size=3, stride=1)),
            ('conv3_2', nn.Conv2d(128, 64, kernel_size=1, stride=1)),
            ('conv3_3', nn.Conv2d(64, 128, kernel_size=3, stride=1)),
            ('pool3', nn.MaxPool2d(2, stride=2)),
            ('conv4_1', nn.Conv2d(128, 256, kernel_size=3, stride=1)),
            ('conv4_2', nn.Conv2d(256, 128, kernel_size=1, stride=1)),
            ('conv4_3', nn.Conv2d(128, 256, kernel_size=3, stride=1)),
            ('pool4', nn.MaxPool2d(2, stride=2)),
            ('conv5_1', nn.Conv2d(256, 512, kernel_size=3, stride=1)),
            ('conv5_2', nn.Conv2d(512, 256, kernel_size=1, stride=1)),
            ('conv5_3', nn.Conv2d(256, 512, kernel_size=3, stride=1)),
            ('conv5_4', nn.Conv2d(512, 256, kernel_size=1, stride=1)),
            ('conv5_5', nn.Conv2d(256, 512, kernel_size=3, stride=1)),
            ('pool5', nn.MaxPool2d(2, stride=2)),
            ('conv6_1', nn.Conv2d(512, 1024, kernel_size=3, stride=1)),
            ('conv6_2', nn.Conv2d(1024, 512, kernel_size=1, stride=1)),
            ('conv6_3', nn.Conv2d(512, 1024, kernel_size=3, stride=1)),
            ('conv6_4', nn.Conv2d(1024, 512, kernel_size=1, stride=1)),
            ('conv6_5', nn.Conv2d(512, 1024, kernel_size=3, stride=1)),
            ('conv7', nn.Conv2d(1024, 1000, kernel_size=1, stride=1)),
            ('pool7', nn.AvgPool2d(kernel_size=2, stride=1)),
        ]
        self.model = nn.Sequential(OrderedDict(network_list))
        self.cls_head = nn.Sequential(OrderedDict([
            ('cls_head_conv', nn.Conv2d(1000, self.anchor_num * self.class_num, kernel_size=1, stride=1)),
            ('cls_head_softmax', nn.Softmax(dim=-1)),
        ]))

        self.reg_head = nn.Sequential(OrderedDict([
            ('reg_head', nn.Conv2d(1000, self.anchor_num * 5, kernel_size=1, stride=1)),
        ]))

    def forward(self, input):
        featuremap = self.model(input)
        cls_score = self.cls_head(featuremap)
        pred_cord = self.reg_head(featuremap)
        return cls_score, pred_cord
