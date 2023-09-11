import torch.utils.data
import torchvision
from loss import YoloLoss
from network import Darknet19
import config
from utils import *
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter(config.tensorboard_logs)

torch.set_default_device(config.default_device)
import transform as T

transform = T.Compose([
    T.ConvertImageToTrainableData(),
    T.ResizeImage(config.input_width, config.input_height),
    T.RandomHorizontalFlip(0.5, config.input_width),
])


def detection_collate(batch):
    """Custom collate fn for dealing with batches of images that have a different
    number of associated object annotations (bounding boxes).
    Arguments:
        batch: (tuple) A tuple of tensor images and lists of annotations
    Return:
        A tuple containing:
            1) (tensor) batch of images stacked on their 0 dim
            2) (list of tensors) annotations for a given image are stacked on
                                 0 dim
    """
    targets = []
    imgs = []
    for sample in batch:
        imgs.append(sample[0])
        targets.append(sample[1])
    return torch.stack(imgs, 0), targets


train_dataset = torchvision.datasets.VOCDetection(root=config.dataroot, year="2012", image_set='train', transforms=transform,
                                                  download=False)
test_dataset = torchvision.datasets.VOCDetection(root=config.dataroot, year="2012", image_set='trainval', transforms=transform,
                                                 download=False)

train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=config.batch_size,
                                               collate_fn=detection_collate, drop_last=True)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=config.batch_size,
                                              collate_fn=detection_collate, drop_last=True)

darknet19 = Darknet19()
if os.path.exists(config.path_to_state_dict):
    print("loading model state dict...")
    darknet19.load_state_dict(torch.load(config.path_to_state_dict))

criterion = YoloLoss()
optimizer = torch.optim.SGD(params=darknet19.parameters(), lr=config.learning_rate, momentum=config.momentum, weight_decay=config.weight_decay)

for epoch in range(config.train_epochs):
    darknet19.train()
    for iter, (image, target) in enumerate(train_dataloader):
        pred = darknet19(image)
        optimizer.zero_grad()
        loss = criterion(iter + epoch * len(train_dataloader), pred, target)
        loss.backward()
        optimizer.step()
        if (iter + epoch * len(train_dataloader)) % config.loss_print_period == 0:
            writer.add_scalar("loss", loss, iter + epoch * len(train_dataloader))
            print("epoch:{} iter:{} loss:{}:".format(epoch, iter, loss.item()))
    # save state_dict every epoch
    torch.save(darknet19.state_dict(), config.path_to_state_dict)
    with torch.no_grad():
        cls_score, pred_object = pred
        cls_score_to_show = cls_score[0, ...]
        image_to_show = image[0]
        pred_object_to_show = pred_object[0, ...]
        MapPredCordBackToInputSize(pred_object_to_show)
        pred_to_show = cls_score_to_show, pred_object_to_show
        target = NMSbyConf(pred_to_show)
        img = DrawWithPred(image_to_show, target)
        writer.add_image("pred_result", img, 1, dataformats='CHW')
        # ShowImageWbnd(image_to_show, target)
writer.close()
