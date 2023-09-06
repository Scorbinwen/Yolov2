import torch
import PIL.Image as Image
import numpy as np
from torchvision.utils import draw_bounding_boxes
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
import torchvision.transforms.functional as F
import torchvision.transforms as transforms
import os

import config

def GetCenterAlignIouBetween(box1, box2):
    x1, y1, w1, h1 = box1
    x2, y2, _, _ = box2
    shift_box1 = torch.Tensor([x2, y2, w1, h1], dtype=torch.float32)
    return GetIouBetween(shift_box1, box2)

# utils.py
def GetIouBetween(box1, box2):
    # expect box1: centeroid(x, y), w, h,
    # expect box2 :centeroid(x, y), w, h,
    # convert centorid x, y to topleft x, y and bottomright x, y
    # make sure box1 and box2 are both of float value
    if not isinstance(box1, torch.Tensor) or not isinstance(box2, torch.Tensor):
        raise TypeError("either box1 or box2 is not of Torch.Tensor type")
    box1 = torch.Tensor(box1, dtype=torch.float32)
    box2 = torch.Tensor(box2, dtype=torch.float32)
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    box2_area = w2 * h2
    box1_area = w1 * h1

    x1_tl = x1 - .5 * w1
    y1_tl = y1 - .5 * h1

    x1_br = x1 + .5 * w1
    y1_br = y1 + .5 * h1

    x2_tl = x2 - .5 * w2
    y2_tl = y2 - .5 * h2

    x2_br = x2 + .5 * w2
    y2_br = y2 + .5 * h2

    tl_x = torch.max(x1_tl, x2_tl)
    tl_y = torch.max(y1_tl, y2_tl)
    br_x = torch.min(x1_br, x2_br)
    br_y = torch.min(y1_br, y2_br)
    overlap = 0
    if (br_x > tl_x) and (br_y > tl_y):
        overlap = (br_x - tl_x) * (br_y - tl_y)
    # ensure that the denominator != 0
    iou = overlap / (box1_area + box2_area - overlap)
    return iou


def show_image_wbnd(image=None, target=None):
    """
    Sample image and boxes
    image = torch.randint(0, 255, (3, 418, 418), dtype=torch.uint8)
    boxes = torch.tensor([[50, 50, 100, 200], [210, 150, 350, 400]], dtype=torch.float)
    """
    # expect shape of image is (3, W, H), and dtype is torch.uint8
    if image.dim == 4:
        image = image[0]
        target = target[0]
    image = image * config.image_normalize_scale
    image = torch.tensor(image, dtype=torch.uint8)
    # image = torch.Tensor(image, dtype=torch.uint8)
    _, _, true_object = target
    boxes = torch.zeros_like(true_object)
    # true_object value is within (0, 13)
    boxes[..., :2] = true_object[..., :2] - (.5 * true_object[..., 2:])
    boxes[..., 2:] = true_object[..., :2] + (.5 * true_object[..., 2:])
    boxes = boxes
    result = draw_bounding_boxes(image, boxes=boxes, width=5)
    trans_to_pil = transforms.ToPILImage(mode="RGB")
    img_pil = trans_to_pil(result)
    img_pil.show()


def write_xml(xml_name, save_name, box, resize_w, resize_h):
    """
    将修改后的box 写入到 xml文件中
    :param xml_name: 原xml
    :param save_name: 保存的xml
    :param box: 修改后需要写入的box
    :return:
    """
    etree = ET.parse(xml_name)
    root = etree.getroot()

    # 修改图片的宽度、高度
    for obj in root.iter('size'):
        obj.find('width').text = str(resize_w)
        obj.find('height').text = str(resize_h)

    # 修改box的值
    for obj, bo in zip(root.iter('object'), box):
        for index, x in enumerate(obj.find('bndbox')):
            x.text = str(int(bo[index]))
    etree.write(save_name)


def save_transformed_data(image, target, root=config.data_root, save_width=config.input_width, save_height=config.input_height):
    origin_target, true_cls, true_object = target
    # show_image_wbnd(image, target)
    true_object[..., :2] = true_object[..., :2] - .5 * true_object[..., 2:]
    true_object[..., 2:] = true_object[..., :2] + .5 * true_object[..., 2:]
    folder = origin_target['annotation']['folder']
    filename = origin_target['annotation']['filename']
    img_subfolder = "ResizedJPEGImages"
    xml_subfolder = "ResizedAnnotations"
    image_path = os.path.join(root, folder, img_subfolder, filename)
    xml_path = os.path.join(root, folder, xml_subfolder, filename.split(".")[0] + ".xml")
    write_xml(xml_path, xml_path, true_object, save_width, save_height)
    image = image.permute(1, 2, 0).cpu()
    image = image.numpy()
    image = (image * config.image_normalize_scale).astype(np.uint8)
    image = Image.fromarray(image)
    image.save(image_path)


def PasteImageToCanvas(image,true_object, dx, dy, canvas, resize_ratio):
    """
    :param image:
    :param dx:
    :param dy:
    :param canvas:
    :return:
    """
    if not isinstance(image, torch.Tensor) or not isinstance(canvas, torch.Tensor):
        raise TypeError("either image or canvas is not of type tensor!")
    _, cvsheight, cvswidth = canvas.shape
    _, imgheight, imgwidth = image.shape

    if (cvsheight - 2 * dy == imgheight) and (cvswidth - 2 * dx == imgwidth):
        canvas[:, dy:cvsheight-dy, dx:cvswidth-dx] = image
    else:  # handle for odd (cvsheight - imgheight) case.
        ddy = cvsheight - 2 * dy - imgheight
        ddx = cvswidth - 2 * dx - imgwidth
        canvas[:, dy:cvsheight-(dy+ddy), dx:cvswidth-(dx+ddx)] = image
    true_object[..., 0] = true_object[..., 0] * resize_ratio + dx
    true_object[..., 1] = true_object[..., 1] * resize_ratio + dy
    true_object[..., 2] = true_object[..., 2] * resize_ratio
    true_object[..., 3] = true_object[..., 3] * resize_ratio
    return canvas, true_object