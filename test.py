from time import time

from utils import AverageMeter, ProgressMeter
from metrics import iou, pixel_acc, pixel_acc2

import torch
from torch.autograd import Variable


def _test(test_loader, model, use_gpu, criterion, n_class, device):
    batch_time = AverageMeter('test time', ':6.3f')
    data_time = AverageMeter('test data time', ':6.3f')
    loss_running = AverageMeter('test Loss', ':.4f')
    miou_running = AverageMeter('test mIoU', ':.4f')
    acc_running = AverageMeter('test acc', ':.4f')
    acc2_running = AverageMeter('test acc2', ':.4f')
    progress = ProgressMeter(
                            len(test_loader),
                            [batch_time, data_time, loss_running, miou_running, acc_running, acc2_running],
                            prefix=f"epoch {0+1} Test")
    with torch.no_grad():
        end = time()
        model.eval()
        for iter, batch in enumerate(test_loader):
            data_time.update(time()-end)
            if use_gpu:
                inputs = Variable(batch['X'].to(device))
                labels = Variable(batch['Y'].to(device))
            else:
                inputs = Variable(batch['X'])
                labels = Variable(batch['Y'])

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            bs = inputs.size(0) # current batch size
            loss = loss.item()
            loss_running.update(loss, bs)

            progress.display(iter)
            batch_time.update(time() - end)
            end = time()

            outputs = pred.data.cpu().numpy()
            N, _, h, w = pred.shape
            pred = pred.transpose(0, 2, 3, 1).reshape(-1, n_class).argmax(axis=1).reshape(N, h, w)

            target = batch['l'].cpu().numpy().reshape(N, h, w)
            for p, t in zip(pred, target):
                acc_running.update(pixel_acc(p, t)) 

            miou_running.update(iou(outputs, labels, n_class))
            acc2_running.update(pixel_acc2(outputs, labels))

            # gpu memory 비우기 
            del batch
            torch.cuda.empty_cache()


    print('\ntest loss {:.4f} | test miou {:.4f} | test acc {:.4f} | test acc2 {:.4f}'.format(loss_running.avg, miou_running.avg, acc_running.avg, acc2_running.avg))
