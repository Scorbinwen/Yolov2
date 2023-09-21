import torch.utils.data
import torchvision
from loss import YoloLoss
from network import *
import config
import signal
from utils import *
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter(config.tensorboard_logs)

torch.set_default_device(config.default_device)
from dataset import *
train_dataloader, test_dataloader = GetDummyDataDataLoader()

model = DarkNet53()
criterion = YoloLoss()
optimizer = torch.optim.RMSprop(params=model.parameters(), lr=config.learning_rate)



def train():
    if os.path.exists(config.path_to_state_dict):
        print("loading model state dict...")
        checkpoint = torch.load(config.path_to_state_dict)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    for epoch in range(config.train_epochs):
        model.train()
        for iter, (image, target) in enumerate(train_dataloader):
            pred = model(image)
            optimizer.zero_grad()
            loss, noobj_loss, obj_loss, prior_loss, true_loss, score_loss = criterion(iter + epoch * len(train_dataloader), pred, target)
            loss.backward()
            optimizer.step()
            if (iter + epoch * len(train_dataloader)) % config.loss_print_period == 0:
                writer.add_scalar("loss", loss, iter + epoch * len(train_dataloader))
                print("epoch:{} iter:{} total_loss:{}: noobj_loss:{}, obj_loss:{}, prior_loss:{}, true_loss:{}, score_loss:{}"
                      "".format(epoch, iter, loss.item(), noobj_loss.item(), obj_loss.item(), prior_loss.item(), true_loss.item(), score_loss.item()))
        # save state_dict every epoch.
        s = signal.signal(signal.SIGINT, signal.SIG_IGN)
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
        }, config.path_to_state_dict)
        signal.signal(signal.SIGINT, s)

        with torch.no_grad():
            cls_score, pred_object = pred
            cls_score_to_show = cls_score[0, ...]
            image_to_show = image[0]
            pred_object_to_show = pred_object[0, ...]
            pred_object_to_show = MapPredCordBackToInputSize(pred_object_to_show)
            pred_to_show = cls_score_to_show, pred_object_to_show
            target = NMSbyConf(pred_to_show)
            img = DrawWithPred(image_to_show, target)
            # writer.add_image("pred_result", img, global_step=None, walltime=None, dataformats='CHW')
            # ShowImageWbnd(img)
    writer.close()


def eval():
    if os.path.exists(config.path_to_state_dict):
        print("loading model state dict...")
        checkpoint = torch.load(config.path_to_state_dict)
        model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    with torch.no_grad():
        for iter, (image, target) in enumerate(test_dataloader):
                print("image shape", image.shape)
                pred = model(image)
                cls_score, pred_object = pred
                cls_score_to_show = cls_score[0, ...]
                image_to_show = image[0]
                pred_object_to_show = pred_object[0, ...]
                pred_object_to_show = MapPredCordBackToInputSize(pred_object_to_show)
                pred_to_show = cls_score_to_show, pred_object_to_show
                target = NMSbyConf(pred_to_show)
                img = DrawWithPred(image_to_show, target)
                # writer.add_image("pred_result", img, global_step=None, walltime=None, dataformats='CHW')
                ShowImageWbnd(img)


eval()