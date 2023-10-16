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

model = Yolov2(trainable=True)
criterion = YoloLoss()
optimizer = torch.optim.SGD(model.parameters(),
                      lr=config.learning_rate,
                      momentum=config.momentum,
                      weight_decay=config.weight_decay
                      )

def UpdateLearningRate(optimizer):
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * 0.1

def set_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def train():
    if os.path.exists(config.path_to_state_dict):
        print("loading model state dict...")
        checkpoint = torch.load(config.path_to_state_dict)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # for name, param in model.named_parameters():
    #     print("name:{} param:{}".format(name, param))
    tmp_lr = config.learning_rate
    base_lr = config.learning_rate
    model.train(True)
    for epoch in range(config.train_epochs):
        if epoch in config.lr_epoch:
            tmp_lr = tmp_lr * 0.1
            set_lr(optimizer, tmp_lr)
        for iter, (image, target) in enumerate(train_dataloader):
            ni = iter+epoch*len(train_dataloader)
            # 使用warm-up策略来调整早期的学习率
            if not config.no_warm_up:
                if epoch < config.wp_epoch:
                    nw = config.wp_epoch*len(train_dataloader)
                    tmp_lr = base_lr * pow((ni)*1. / (nw), 4)
                    set_lr(optimizer, tmp_lr)
                    print("tmp_lr", tmp_lr)
                elif epoch == config.wp_epoch and iter == 0:
                    tmp_lr = base_lr
                    set_lr(optimizer, tmp_lr)
            print("tmp_lr", tmp_lr)
            pred = model(image)
            loss, noobj_loss, obj_loss, score_loss, true_loss_xy, true_loss_wh = criterion(
                iter + epoch * len(train_dataloader), pred, target)
            loss.backward()
            # for name, v in model.named_parameters():
            #     if name == "backbone.conv_1.0.convs.0.weight":
            #         print("name:{} grad:{}".format(name, v.grad))
            optimizer.step()
            optimizer.zero_grad()
            if (iter + epoch * len(train_dataloader)) % config.loss_print_period == 0:
                writer.add_scalar("loss", loss, iter + epoch * len(train_dataloader))
                print("epoch:{} iter:{} total_loss:{}: noobj_loss:{}, obj_loss:{}, score_loss:{}, true_loss_xy:{}, "
                      "true_loss_wh:{}".
                      format(epoch, iter, loss.item(), noobj_loss.item(), obj_loss.item(),
                             score_loss.item(), true_loss_xy.item(), true_loss_wh.item()))
        # UpdateLearningRate(optimizer)
        # save state_dict every epoch.
        s = signal.signal(signal.SIGINT, signal.SIG_IGN)
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
        }, config.path_to_state_dict)
        signal.signal(signal.SIGINT, s)

        # with torch.no_grad():
        #     img = DrawWithPredResult(pred, image)
        #     # writer.add_image("pred_result", img, global_step=None, walltime=None, dataformats='CHW')
        #     ShowImageWbnd(img)
    writer.close()


def eval():
    if os.path.exists(config.path_to_state_dict):
        print("loading model state dict...")
        checkpoint = torch.load(config.path_to_state_dict)
        model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    with torch.no_grad():
        for iter, (image, target) in enumerate(test_dataloader):
            with torch.no_grad():
                pred = model(image)
                img = DrawWithPredResult(pred, image)
                # writer.add_image("pred_result", img, global_step=None, walltime=None, dataformats='CHW')
                ShowImageWbnd(img)

eval()
