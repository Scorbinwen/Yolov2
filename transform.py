import torch
from torchvision.transforms import functional as F
import torchvision.transforms as T
import numpy as np
from utils import show_image_wbnd
from utils import PasteImageToCanvas
import config

class Compose(object):
    def __init__(self, transform_list):
        if not isinstance(transform_list, list):
            raise TypeError("expect transform_list type is list!")
        self.transform_list = transform_list

    def __call__(self, image, target):
        for t in self.transform_list:
            image, target = t(image, target)
        return image, target


class ConvertImageToTrainableData(object):
    def __init__(self):
        self.NameToClsId = config.NameToClsId
        self.cls_num = config.class_num

    def __call__(self, image, target):
        image = torch.from_numpy(np.array(image)).permute(2, 0, 1)
        if not isinstance(image, torch.Tensor):
            raise TypeError("expect image is of type Torch.Tensor!")
        # normalize image to be of value 0~1
        image = image / config.image_normalize_scale
        object_num = len(target["annotation"]["object"])
        true_label = torch.zeros(object_num, self.cls_num)
        true_object = torch.zeros(object_num, 4)
        for ind in range(object_num):
            true_label[ind, self.NameToClsId[target["annotation"]["object"][ind]["name"]]] = 1
            xmin = float(target["annotation"]["object"][ind]["bndbox"]["xmin"])
            ymin = float(target["annotation"]["object"][ind]["bndbox"]["ymin"])
            xmax = float(target["annotation"]["object"][ind]["bndbox"]["xmax"])
            ymax = float(target["annotation"]["object"][ind]["bndbox"]["ymax"])
            true_object[ind, 0] = .5 * (xmin + xmax)
            true_object[ind, 1] = .5 * (ymin + ymax)
            true_object[ind, 2] = xmax - xmin
            true_object[ind, 3] = ymax - ymin
        # keep origin target for retrieve origin information with little memory
        target_trainable = (target, true_label, true_object)
        return image, target_trainable


class RandomHorizontalFlip(object):
    def __init__(self, flip_prob=config.flip_prob, fm_width=config.output_width):
        self.flip_prop = flip_prob
        self.fm_width = fm_width

    def __call__(self, image, target):
        origin_target, true_class, true_object = target
        if torch.rand(1) < self.flip_prop:
            image = F.hflip(image)
            true_object[..., 0] = self.fm_width - true_object[..., 0]
        target = (origin_target, true_class, true_object)
        return image, target


class ResizeImage(object):
    """
    we want to keep the W-H ratio of origin image when resize image,
    so that the network can better learning features that is not warped,
    of course we can use deformable convolution, but that increase implement complexity.
    """
    def __init__(self, *size):
        self.theight, self.twidth = size

    def __call__(self, image, target):
        origin_target, true_class, true_object = target
        if not isinstance(image, torch.Tensor):
            raise TypeError ("expect image type is torch.Tensor!")
        if image.shape[0] == 3:
            sheight, swidth = image.shape[1:]
        else:
            sheight, swidth = image.shape[:2]

        black_canvas = torch.zeros(3, self.theight, self.twidth)
        resizeh_ratio = self.theight / sheight
        resizew_ratio = self.twidth / swidth

        if resizew_ratio * sheight <= self.theight:
            resize_ratio = resizew_ratio
        else:
            resize_ratio = resizeh_ratio

        resized_height = int(resize_ratio * sheight)
        resized_width = int(resize_ratio * swidth)

        dy = (self.theight - resized_height) // 2
        dx = (self.twidth - resized_width) // 2

        resize_ops = T.Resize((resized_height, resized_width))
        resized_image = resize_ops(image)
        image, true_object = PasteImageToCanvas(resized_image, true_object, dx, dy, black_canvas, resize_ratio)

        target = (origin_target, true_class, true_object)
        # show_image_wbnd(image, target)
        return image, target
