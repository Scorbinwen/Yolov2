
NameToClsId = \
    {'aeroplane': 0, 'bicycle': 1, 'bird': 2, 'boat': 3, 'bottle': 4,
    'bus': 5, 'car': 6, 'cat': 7, 'chair': 8, 'cow': 9,
    'diningtable': 10, 'dog': 11, 'horse': 12, 'motorbike': 13, 'person': 14,
    'pottedplant': 15, 'sheep': 16, 'sofa': 17, 'train': 18, 'tvmonitor': 19}
batch_size = 1
input_width = 418
input_height = 418
class_num = 20
anchor_num = 4
output_width = 13
output_height = 13
anchor_num=4
class_num=20
iou_threshold=0.6
scale_noobj=0.05
scale_obj = 1.0
image_normalize_scale = 256
flip_prob=0.5
data_root = "data\\VOCdevkit"
learning_rate = 1e-4
anchor_box = \
    [[0.08285376, 0.13705531],
    [0.20850361, 0.39420716],
    [0.80552421, 0.77665105],
    [0.42194719, 0.62385487]]
anchor_train_iters = 12800