import torch

import config

NameToClsId = \
    {'aeroplane': 0, 'bicycle': 1, 'bird': 2, 'boat': 3, 'bottle': 4,
     'bus': 5, 'car': 6, 'cat': 7, 'chair': 8, 'cow': 9,
     'diningtable': 10, 'dog': 11, 'horse': 12, 'motorbike': 13, 'person': 14,
     'pottedplant': 15, 'sheep': 16, 'sofa': 17, 'train': 18, 'tvmonitor': 19}

ClsIdToName = \
    {0: 'aeroplane', 1: 'bicycle', 2: 'bird', 3: 'boat', 4: 'bottle',
     5: 'bus', 6: 'car', 7: 'cat', 8: 'chair', 9: 'cow',
     10: 'diningtable', 11: 'dog', 12: 'horse', 13: 'motorbike', 14: 'person',
     15: 'pottedplant', 16: 'sheep', 17: 'sofa', 18: 'train', 19: 'tvmonitor'}
class_num = 2 + 1  # 1 denotes no object
batch_size = 8
test_batch_size = 1
input_size = 416
anchor_num = 5
output_size = 13
iou_threshold = 0.5
scale_noobj = 0.2
scale_obj = 1.0
image_normalize_scale = 256
flip_prob = 0.5
data_root = "data\\VOCdevkit"
dataroot = "data"
learning_rate = 1e-4
is_dummydata = True

if is_dummydata:
    dummy_lower_limit = 100
    dummy_upper_limit = 120
    dummy_dataset_len = 5000
    step = (dummy_upper_limit - dummy_lower_limit) / 32 / 4
    start_size = dummy_lower_limit / 32
    anchor_box = \
        [[start_size, start_size],
         [start_size + step, start_size + step],
         [start_size + 2 * step, start_size + 2 * step],
         [start_size + 3 * step, start_size + 3 * step],
         [start_size + 4 * step, start_size + 4 * step]]
else:
    anchor_box = \
        [[0.08285376, 0.13705531],
         [0.20850361, 0.39420716],
         [0.80552421, 0.77665105],
         [0.42194719, 0.62385487]]

anchor_train_epochs = 3
# note that we assume the input_size equals to input_size
downsample_rate = input_size // output_size
loss_print_period = 10
nms_iou_threshold = 0.5

x = torch.arange(0, output_size, requires_grad=False, device="cuda")
y = torch.arange(0, output_size, requires_grad=False, device="cuda")
x_cord, y_cord = torch.meshgrid(x, y)
fm_cord = torch.concat((x_cord[..., None], y_cord[..., None]), dim=-1)

path_to_state_dict = 'ModelPth\\state_dict_model.pth'
loss_print_period = 10
default_device = "cuda"
tensorboard_logs = './logs'
train_epochs = 50
lr_epoch=[100, 150]
wp_epoch=1
weight_decay = 5e-4
momentum = 0.9

train_detection = True
show_pred_every_iter = False
head_dim = 1024
no_warm_up=False
reorg_dim=64

pretrained_backbone=True
backbone_state_dict = "ModelPth\\darknet19.pth"