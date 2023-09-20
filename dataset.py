import torch.nn.functional
from torch.utils.data import Dataset
import cv2
import config
import numpy as np
import random
from utils import *
import transform as T
import torchvision


class DummyDataset(Dataset):
    def __init__(self):
        pass

    def __len__(self):
        return config.dummy_dataset_len

    def __getitem__(self, index):
        counters = np.zeros((config.input_height, config.input_width, 3), dtype=np.uint8)

        target_cls = []
        target_object = []

        pick_circle_x, pick_circle_y, pick_circle_size, iouflag = RandomGenerateBbox(target_object)
        if iouflag == False:
            cv2.circle(counters, (pick_circle_x, pick_circle_y), pick_circle_size // 2, (255, 255, 0), -1)
            target_object.append(torch.tensor([pick_circle_x, pick_circle_y, pick_circle_size, pick_circle_size]))
            target_cls.append(torch.tensor(0))

        pick_rect_x, pick_rect_y, pick_rect_size, iouflag = RandomGenerateBbox(target_object)
        if iouflag == False:
            cv2.rectangle(counters, (pick_rect_x - pick_rect_size // 2, pick_rect_y - pick_rect_size // 2),
                          (pick_rect_x + pick_rect_size // 2, pick_rect_y + pick_rect_size // 2), (125, 0, 125), -1)
            target_object.append(torch.tensor([pick_rect_x, pick_rect_y, pick_rect_size, pick_rect_size]))
            target_cls.append(torch.tensor(1))

        # transform image & target
        target_cls = torch.stack(target_cls)
        target_object = torch.stack(target_object)
        target = None, torch.nn.functional.one_hot(target_cls, num_classes=config.class_num), target_object

        image = torch.tensor(counters) / 256
        image = image.permute(2, 0, 1)
        # obj_target = target_cls, target_object
        # ShowImageWbnd(image, obj_target)
        return image, target


transform = T.Compose([
    T.ConvertImageToTrainableData(),
    T.ResizeImage(config.input_width, config.input_height),
    T.RandomHorizontalFlip(0.5, config.input_width),
])


def detection_collate(batch):
    targets = []
    imgs = []
    for sample in batch:
        imgs.append(sample[0])
        targets.append(sample[1])
    return torch.stack(imgs, 0), targets

def GetVOCDetectionDataLoader():
    train_dataset = torchvision.datasets.VOCDetection(root=config.dataroot, year="2012", image_set='train', transforms=transform,
                                                      download=False)
    test_dataset = torchvision.datasets.VOCDetection(root=config.dataroot, year="2012", image_set='trainval', transforms=transform,
                                                     download=False)

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=config.batch_size,
                                                   collate_fn=detection_collate, drop_last=True)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=config.batch_size,
                                                  collate_fn=detection_collate, drop_last=True)
    return train_dataloader, test_dataloader


def GetDummyDataDataLoader():
    # 数据集实例
    train_dataset = DummyDataset()
    test_dataset = DummyDataset()

    # 数据迭代器
    train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=config.batch_size,
                                                   collate_fn=detection_collate, drop_last=True)
    test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=config.batch_size,
                                                   collate_fn=detection_collate, drop_last=True)
    return train_dataloader, test_dataloader