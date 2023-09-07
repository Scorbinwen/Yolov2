
import torch.utils.data
import torchvision
from utils import show_image_wbnd
from loss import YoloLoss
from network import Darknet19
import config
from utils import save_transformed_data

torch.set_default_device("cuda")


import transform as T
transform = T.Compose([
    T.ConvertImageToTrainableData(),
    T.ResizeImage(config.input_width, config.input_height),
    #T.RandomHorizontalFlip(0.5, 418),
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

root = "data"
train_dataset = torchvision.datasets.VOCDetection(root=root,  year="2012", image_set='train', transforms=transform, download=False)
test_dataset = torchvision.datasets.VOCDetection(root=root,  year="2012", image_set='trainval', transforms=transform, download=False)

train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=config.batch_size, collate_fn=detection_collate, drop_last=True)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=config.batch_size, collate_fn=detection_collate, drop_last=True)


darknet19 = Darknet19()
criterion = YoloLoss()
learning_rate = config.learning_rate
optimizer = torch.optim.RMSprop(params=darknet19.parameters(), lr=learning_rate)



for epoch in range(1):
    darknet19.train()
    for iter, (image, target) in enumerate(train_dataloader):
        # show_image_wbnd(image[0], target[0])
        # save_transformed_data(image[0], target[0])
        # print(iter)
        pred = darknet19(image)
        # print("pre shape", pred[0].shape, pred[1].shape)
        # print("target shape", len(target[0]), len(target[1]))
        optimizer.zero_grad()
        loss = criterion(iter + epoch * len(train_dataloader), pred, target)
        loss.backward()
        optimizer.step()
        if (iter % config.loss_print_period == 0):
            print("iter:{} loss:{}:".format(iter, loss.item()))
    with torch.no_grad():
        darknet19.eval()
        for iter, (image, target) in enumerate(test_dataloader):
            pred = darknet19(image)






