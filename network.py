import torch.nn as nn
from collections import OrderedDict
import config
import torch
import numpy as np
import torch.nn.functional as F


class ConvolutionalLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernal_size, stride, padding):
        super(ConvolutionalLayer, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernal_size, stride, padding),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1)
        )

    def forward(self, x):
        return self.conv(x)


# 残差块结构
class ResidualLayer(nn.Module):
    def __init__(self, in_channels):
        super(ResidualLayer, self).__init__()
        self.reseblock = nn.Sequential(
            ConvolutionalLayer(in_channels, in_channels // 2, kernal_size=1, stride=1, padding=0),
            ConvolutionalLayer(in_channels // 2, in_channels, kernal_size=3, stride=1, padding=1)
        )

    def forward(self, x):
        return x + self.reseblock(x)


class wrapLayer(nn.Module):
    def __init__(self, in_channels, count):
        super(wrapLayer, self).__init__()
        self.count = count
        self.in_channels = in_channels
        self.res = ResidualLayer(self.in_channels)

    def forward(self, x):
        for i in range(0, self.count):
            x = self.res(x)
        return x


class DownSampleLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownSampleLayer, self).__init__()
        self.conv = nn.Sequential(
            ConvolutionalLayer(in_channels, out_channels, kernal_size=3, stride=2, padding=1)
        )

    def forward(self, x):
        return self.conv(x)


class DarkNet53(nn.Module):
    def __init__(self):
        super(DarkNet53, self).__init__()
        self.class_num = config.class_num
        self.anchor_num = config.anchor_num
        self.fm_width = config.output_size
        self.fm_height = config.output_size
        self.feature_52 = nn.Sequential(
            ConvolutionalLayer(3, 32, 3, 1, 1),
            DownSampleLayer(32, 64),
            ResidualLayer(64),
            DownSampleLayer(64, 128),
            wrapLayer(128, 2),
            DownSampleLayer(128, 256),
            wrapLayer(256, 8)
        )
        self.feature_26 = nn.Sequential(
            DownSampleLayer(256, 512),
            wrapLayer(512, 8)
        )
        self.feature_13 = nn.Sequential(
            DownSampleLayer(512, 1024),
            wrapLayer(1024, 4)
        )
        self.cls_head = nn.Sequential(
            nn.Conv2d(1024, self.anchor_num * self.class_num, kernel_size=1, stride=1),
        )

        self.reg_head = nn.Sequential(
            nn.Conv2d(1024, self.anchor_num * 5, kernel_size=1, stride=1),
        )
        self.sigmoid = torch.nn.Sigmoid()
        self.softmax = torch.nn.Softmax(dim=-1)
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.conv = nn.Conv2d(1024, config.class_num, kernel_size=1, stride=1)

    def forward(self, x):
        h_52 = self.feature_52(x)
        h_26 = self.feature_26(h_52)
        h_13 = self.feature_13(h_26)

        cls_out = self.conv(h_13)
        cls_out = self.global_avg_pool(cls_out)
        cls_out = cls_out.view(-1, config.class_num)
        cls_out = self.softmax(cls_out)

        cls_score = self.cls_head(h_13)
        cls_score = torch.reshape(cls_score, (-1, self.fm_width, self.fm_height, self.anchor_num, self.class_num))
        cls_score = self.softmax(cls_score)
        pred_object = self.reg_head(h_13)
        pred_object = torch.reshape(pred_object, (-1, self.fm_width, self.fm_height, self.anchor_num, 5))

        pred_object[..., :2] = self.sigmoid(pred_object[..., :2]) + config.fm_cord[..., None, :2]
        fm_center = config.output_size / 2
        with torch.no_grad():
            fm_size_limit = 2 * (fm_center - torch.abs(pred_object[..., :2] - fm_center))

        pred_object[..., 2:4] = self.sigmoid(pred_object[..., 2:4]) * fm_size_limit[..., :2]
        pred_object[..., 4] = self.sigmoid(pred_object[..., 4])

        index = torch.where((pred_object[..., 0] >= 4) & (pred_object[..., 0] <= 8) & (pred_object[..., 1] >= 4) & (
                    pred_object[..., 1] <= 8))
        print("pred_object", pred_object[index])
        return cls_out, cls_score, pred_object
