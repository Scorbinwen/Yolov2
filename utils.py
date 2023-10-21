import torch
import PIL.Image as Image
import numpy as np
from torchvision.utils import draw_bounding_boxes
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
import torchvision.transforms.functional as F
import torchvision.transforms as transforms
import os
import random

import config


def GetCenterAlignIouBetween(box1, box2):
    x1, y1, w1, h1 = box1
    x2, y2, _, _ = box2
    shift_box1 = torch.tensor([x2, y2, w1, h1], dtype=torch.float32)
    return GetIouBetween(shift_box1, box2)


# utils.py
def GetIouBetween(box1, box2):
    # expect box1: centeroid(x, y), w, h,
    # expect box2 :centeroid(x, y), w, h,
    # convert centorid x, y to topleft x, y and bottomright x, y
    # make sure box1 and box2 are both of float value
    box1 = torch.tensor(box1, dtype=torch.float32)
    box2 = torch.tensor(box2, dtype=torch.float32)
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


def DrawWithPred(image, target):
    image = image * config.image_normalize_scale
    image = torch.tensor(image, dtype=torch.uint8)
    # image = torch.Tensor(image, dtype=torch.uint8)
    true_class, true_object = target
    if len(true_object) == 0:
        return image
    boxes = torch.zeros_like(true_object[..., :4])
    # print("true_object1", true_object[..., :2].shape)
    # print("true_object2", true_object[..., 2:4].shape)
    # true_object value is within (0, 13)
    boxes[..., :2] = true_object[..., :2] - (.5 * true_object[..., 2:4])
    boxes[..., 2:] = true_object[..., :2] + (.5 * true_object[..., 2:4])
    # print(boxes)
    labels = []
    for cls_ind in true_class:
        labels.append(config.ClsIdToName[cls_ind.item()])
    result = draw_bounding_boxes(image, boxes=boxes, labels=labels, colors=(255, 0, 0), width=5)
    return result


def ShowImageWbnd(image):
    trans_to_pil = transforms.ToPILImage(mode="RGB")
    img_pil = trans_to_pil(image)
    img_pil.show()


def WriteXml(xml_name, save_name, box, resize_w, resize_h):
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


def SaveTransformedData(image, target):
    origin_target, true_cls, true_object = target
    # ShowImageWbnd(image, target)
    true_object[..., :2] = true_object[..., :2] - .5 * true_object[..., 2:]
    true_object[..., 2:] = true_object[..., :2] + .5 * true_object[..., 2:]
    folder = origin_target['annotation']['folder']
    filename = origin_target['annotation']['filename']
    img_subfolder = "ResizedJPEGImages"
    xml_subfolder = "ResizedAnnotations"
    image_path = os.path.join(config.data_root, folder, img_subfolder, filename)
    xml_path = os.path.join(config.data_root, folder, xml_subfolder, filename.split(".")[0] + ".xml")
    WriteXml(xml_path, xml_path, true_object, config.input_size, config.input_size)
    image = image.permute(1, 2, 0).cpu()
    image = image.numpy()
    image = (image * config.image_normalize_scale).astype(np.uint8)
    image = Image.fromarray(image)
    image.save(image_path)


def PasteImageToCanvas(image, true_object, dx, dy, canvas, resize_ratio):
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
        canvas[:, dy:cvsheight - dy, dx:cvswidth - dx] = image
    else:  # handle for odd (cvsheight - imgheight) case.
        ddy = cvsheight - 2 * dy - imgheight
        ddx = cvswidth - 2 * dx - imgwidth
        canvas[:, dy:cvsheight - (dy + ddy), dx:cvswidth - (dx + ddx)] = image
    true_object[..., 0] = true_object[..., 0] * resize_ratio + dx
    true_object[..., 1] = true_object[..., 1] * resize_ratio + dy
    true_object[..., 2] = true_object[..., 2] * resize_ratio
    true_object[..., 3] = true_object[..., 3] * resize_ratio
    return canvas, true_object


def SortByConf(same_cls_object):
    same_cls_conf = same_cls_object[..., 4]
    sorted_index = torch.argsort(same_cls_conf, dim=-1, descending=True)
    same_cls_object = same_cls_object[sorted_index]
    return same_cls_object


def RmvAtIndex(object_list, index):
    return torch.cat((object_list[:index, ...], object_list[(index + 1):, ...]), dim=0)


def NMSbyConf(pred, target):
    cls_score, pred_object = pred
    _, true_label, true_object = target

    object_score = pred_object[..., 4]
    # print("torch.softmax(cls_score, dim=-1)", torch.softmax(cls_score, dim=-1))
    # print("cls_score max", torch.max(torch.softmax(cls_score, dim=-1), dim=-1))
    cls_score, labels = torch.max(torch.softmax(cls_score, dim=-1), dim=-1)
    # (H x W x KA,)

    pred_scores = (cls_score * object_score).flatten()
    pred_cord = pred_object[..., :4].view(-1, 4)

    pred_prob, topk_idxs = pred_scores.sort(descending=True)
    topk = min(config.topk, pred_scores.size(0))

    pred_cord = pred_cord[topk_idxs][:topk, ...]
    # print("pred_cord", pred_cord)
    cls_score = cls_score.flatten()
    cls_score = cls_score[topk_idxs]
    labels = labels.flatten()
    labels = labels[topk_idxs]
    cls_score = cls_score[:topk, ...]
    object_score = object_score.flatten()
    object_score = object_score[topk_idxs]
    object_score = object_score[:topk, ...]
    pred_prob = pred_prob[:topk, ...]

    candiate_index = torch.where(pred_prob > config.score_threshold)
    pred_cord = pred_cord[candiate_index]
    labels = labels[candiate_index]
    pred_prob = pred_prob[candiate_index]
    object_score = object_score[candiate_index]
    cls_score = cls_score[candiate_index]

    result_bbox = torch.empty(0, 4)
    result_class = torch.empty(0, 1)
    result_prob = torch.empty(0, 1)
    result_object_score = torch.empty(0, 1)
    result_cls_score = torch.empty(0, 1)
    for cls_ind in range(config.class_num - 1):
        same_cls_index = torch.where(labels == cls_ind)
        # print("same_cls_index", same_cls_index)
        # object tensor list of the same class
        same_cls_bbox = pred_cord[same_cls_index]
        same_cls_pred_prob = pred_prob[same_cls_index]
        same_cls_object_score = object_score[same_cls_index]
        same_cls_cls_score = cls_score[same_cls_index]

        # 得到相同类别的cherry_pick的index
        rest_indexs = [i for i in range(0, same_cls_bbox.size(0))]
        cherry_pick_indexs = []
        while len(rest_indexs) > 0:
            cherry_pick_bbox = same_cls_bbox[rest_indexs[0]]
            cherry_pick_indexs.append(rest_indexs[0])

            loop_same_cls_bbox = same_cls_bbox[rest_indexs]
            remove_indexs = []
            for i, bbox in enumerate(loop_same_cls_bbox):
                iou = GetIouBetween(cherry_pick_bbox, bbox)
                if iou > config.nms_iou_threshold:
                    remove_indexs.append(i)
            rest_indexs = list(filter(lambda ind: ind not in remove_indexs, rest_indexs))

        cherry_pick_same_cls_box = same_cls_bbox[cherry_pick_indexs]
        result_bbox = torch.cat((result_bbox, cherry_pick_same_cls_box))
        result_class = torch.cat((result_class, torch.ones(cherry_pick_same_cls_box.shape[:-1]+(1,)) * cls_ind))
        result_prob = torch.cat((result_prob, same_cls_pred_prob[cherry_pick_indexs].unsqueeze(dim=-1)))
        result_object_score = torch.cat((result_object_score, same_cls_object_score[cherry_pick_indexs].unsqueeze(dim=-1)))
        result_cls_score = torch.cat((result_cls_score, same_cls_cls_score[cherry_pick_indexs].unsqueeze(dim=-1)))


    result = torch.cat((result_bbox, result_class, result_prob, result_object_score, result_cls_score), dim=-1)
    print("result", result)
    print("result // 32", result // 32)
    if len(result_bbox) == 0:
        return torch.tensor([]), torch.tensor([])

    return result_class, result_bbox


def MapPredCordBackToInputSize(pred_object):
    pred_object[..., :2] = pred_object[..., :2].flip(dims=[-1])
    pred_object[..., :4] = pred_object[..., :4] * config.downsample_rate
    return pred_object

def RandomGenerateBbox(object_list):
    # bbox size range from 1/4 * 416 = 104  to 3/8 * 416 = 156
    # randomly generate circle
    # randomly pick a size
    pick_size = random.randint(config.dummy_lower_limit, config.dummy_upper_limit)
    # pick_size = 230
    pick_x = random.randint(pick_size // 2, config.input_size - pick_size // 2)
    pick_y = random.randint(pick_size // 2, config.input_size - pick_size // 2)

    # pick_x = random.randint(192, 224)
    # pick_y = random.randint(192, 224)
    # pick_x = 240
    # pick_y = 220
    iouflag = False
    # newly generated bbox should not be overlapped with previous bboxes
    for bbox in object_list:
        iou = GetIouBetween([pick_x, pick_y, pick_size, pick_size], bbox)
        if iou != 0:
            iouflag = True
            break
    return pick_x, pick_y, pick_size, iouflag


def GetTargetToShow(target):
    _, true_label, true_object = target
    _true_object = torch.ones(1, 5)
    _true_object[..., :4] = true_object
    return true_label, _true_object


def DrawWithPredResult(pred, image, target):
    conf, pred_xy, pred_wh, cls_score, cls_out = pred
    _, true_label, true_object = target[0]
    cls_score_to_show = cls_score[0, ...]
    image_to_show = image[0]
    pred_object_to_show = torch.cat((pred_xy[0, ...], pred_wh[0, ...], conf[0, ...]), dim=-1)
    pred_object_to_show = MapPredCordBackToInputSize(pred_object_to_show)
    pred_to_show = cls_score_to_show, pred_object_to_show
    result = NMSbyConf(pred_to_show, target[0])
    img = DrawWithPred(image_to_show, result)
    return img