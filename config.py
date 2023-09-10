import torch

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
class_num = len(NameToClsId) + 1  # 1 denotes no object
batch_size = 16
input_width = 418
input_height = 418
class_num = 20
anchor_num = 4
output_width = 13
output_height = 13
anchor_num = 4
iou_threshold = 0.6
scale_noobj = 0.05
scale_obj = 1.0
image_normalize_scale = 256
flip_prob = 0.5
data_root = "data\\VOCdevkit"
dataroot = "data"
learning_rate = 1e-4
anchor_box = \
    [[0.08285376, 0.13705531],
     [0.20850361, 0.39420716],
     [0.80552421, 0.77665105],
     [0.42194719, 0.62385487]]
anchor_train_iters = 12800
# note that we assume the input_width equals to input_height
downsample_rate = input_width // output_width
loss_print_period = 10
nms_iou_threshold = 0.7

x = torch.arange(0, output_width, requires_grad=False, device="cuda")
y = torch.arange(0, output_height, requires_grad=False, device="cuda")
x_cord, y_cord = torch.meshgrid(x, y)
fm_cord = torch.concat((x_cord[..., None], y_cord[..., None]), dim=-1)
path_to_state_dict = 'ModelPth\\state_dict_model.pth'
loss_print_period = 10
default_device = "cuda"
tensorboard_logs = './logs'