# -*- coding: utf-8 -*-

from __future__ import print_function

import torch
from torch.autograd import Variable

from utils import AverageMeter, ProgressMeter
from metrics import iou, pixel_acc, pixel_acc2

from time import time

def train(epochs, train_loader, val_loader, model, optimizer, use_gpu, criterion, dataset,
          n_class, metrics, scheduler, writer_train, writer_val, early_stopping, device):

    since = time()
    best_miou = 0.0

    for epoch in range(epochs):
        print(f"\n---------------------------------------------------------------------------------------------------------------------\nEpoch {epoch+1}")
        batch_time = AverageMeter('train time', ':6.3f')
        data_time = AverageMeter('train data time', ':6.3f')
        loss_running = AverageMeter('train loss', ':.4f')
        miou_running = AverageMeter('train mIoU', ':.4f')
        p_acc_running = AverageMeter('train P_acc', ':.4f')
        p_acc2_running = AverageMeter('train P_acc2', ':.4f')
        progress = ProgressMeter(
                                 len(train_loader),
                                 [batch_time, data_time, loss_running,
                                  miou_running, p_acc_running, p_acc2_running],
                                 prefix=f"epoch {epoch+1} Train ")
        
        # set model in training mode 
        model.train()
        end = time()

        with torch.set_grad_enabled(True): # 학습 시에만 연산 기록 추적
            for iter, batch in enumerate(train_loader):
                data_time.update(time()-end)
                optimizer.zero_grad()

                if use_gpu:
                    inputs = Variable(batch['X'].to(device))
                    labels = Variable(batch['Y'].to(device))
                    if dataset == "semantic_kitti":
                        inputs_rem = Variable(batch['X_rem'].to(device))
                        labels_rem = Variable(batch['Y_rem'].to(device))

                else:
                    inputs, labels,  = Variable(batch['X']), Variable(batch['Y'])
                    if dataset == "semantic_kitti":
                        inputs_rem, labels_rem = Variable(batch['X_rem']), Variable(batch['Y_rem'])

                if dataset == "semantic_kitti":
                    outputs = model(inputs, inputs_rem)
                else:
                    outputs = model(inputs) # 3 96 320
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                # statistics
                bs = inputs.size(0) # current batch size
                loss = loss.item()
                loss_running.update(loss, bs)

                # output training info
                progress.display(iter)

                # Measure time
                batch_time.update(time()-end)
                end = time()

                pred = outputs.data.cpu().numpy()
                N, _, h, w = pred.shape
                pred = pred.transpose(0, 2, 3, 1).reshape(-1, n_class).argmax(axis=1).reshape(N, h, w)

                target = batch['l'].cpu().numpy().reshape(N, h, w)
                for p, t in zip(pred, target):
                    p_acc_running.update(pixel_acc(p, t)) 

                miou_running.update(iou(outputs, labels, n_class, ignore_class=0))
                p_acc2_running.update(pixel_acc2(outputs, labels, ignore_class=0))
                
                # gpu memory 비우기 
                del batch
                torch.cuda.empty_cache()

        train_loss = loss_running.avg
        train_miou = miou_running.avg
        train_acc = p_acc_running.avg
        train_acc2 = p_acc2_running.avg

        metrics['train_loss'].append(train_loss)
        metrics['train_miou'].append(train_miou)
        metrics['train_acc'].append(train_acc)         
        metrics['train_acc2'].append(train_acc2)
        print('\ntrain loss {:.4f} | train miou {:.4f} | train acc {:.4f} | train acc2 {:.4f}'.format(train_loss, train_miou, train_acc, train_acc2))

        val_loss, val_miou, val_acc, val_acc2  = val(model, criterion, epoch, val_loader, use_gpu, n_class, device)
        metrics['val_loss'].append(val_loss)
        metrics['val_miou'].append(val_miou)
        metrics['val_acc'].append(val_acc)
        metrics['val_acc2'].append(val_acc2)

        scheduler.step()

        # save history 
        with open('./log/result.csv', 'a') as epoch_log:
            epoch_log.write('{} \t\t {:.4f} \t {:.4f} \t {:.4f} \t {:.4f} \t {:.4f} \t {:.4f} \t {:.4f} \t {:.4f}\n'.format(epoch, train_loss, val_loss, train_miou, val_miou, train_acc, val_acc, train_acc2, val_acc2))

        # save model per epochs         --------------------------------------------------
        torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(), # Encoder Decoder 따로 저장을 고려할 때 더 자세히 파보면 가능할 것 같다. (전지연)
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_miou': best_miou,
                    'metrics': metrics,
                    }, './weights/last_weights.pth.tar')

        # Save best miou model to file       --------------------------------------------------
        if val_miou > best_miou:
            print('mIoU improved from {:.4f} to {:.4f}.'.format(best_miou, val_miou))
            best_miou = val_miou
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_miou': best_miou,
                'metrics': metrics}, './weights/best_miou_weights.pth.tar')
        
        # early stopping                --------------------------------------------------
        early_stopping(val_loss=val_loss, model=model, epoch=epoch, optimizer=optimizer, best_miou=best_miou, metrics=metrics)
        if early_stopping.early_stop:
            break

        # tensorboard                   --------------------------------------------------
        writer_train.add_scalar("Loss", train_loss, epoch)
        writer_train.add_scalar("mIoU", train_miou, epoch)
        writer_train.add_scalar("pixel_acc", train_acc, epoch)
        writer_train.add_scalar("pixel_acc2", train_acc2, epoch)
        writer_val.add_scalar("Loss", val_loss, epoch)
        writer_val.add_scalar("mIoU", val_miou, epoch)
        writer_val.add_scalar("pixel_acc", val_acc, epoch)
        writer_val.add_scalar("pixel_acc2", val_acc2, epoch)
        writer_train.flush()
        writer_train.close()
        writer_val.flush()
        writer_val.close()

        time_elapsed = time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

def val(model, criterion, epoch, val_loader, use_gpu, n_class, device):
    batch_time = AverageMeter('val time', ':6.3f')
    data_time = AverageMeter('val data time', ':6.3f')
    loss_running = AverageMeter('val Loss', ':.4f')
    miou_running = AverageMeter('val mIoU', ':.4f')
    acc_running = AverageMeter('val P_acc', ':.4f')
    acc2_running = AverageMeter('val P_acc2', ':.4f')
    progress = ProgressMeter(
                             len(val_loader),
                             [batch_time, data_time, loss_running, miou_running, acc_running, acc2_running],
                             prefix=f"epoch {epoch+1} Test")
    model.eval()
    with torch.no_grad():
        end = time()
        for iter, batch in enumerate(val_loader):
            data_time.update(time()-end)
            if use_gpu:
                inputs = Variable(batch['X'].to(device))
                labels = Variable(batch['Y'].to(device))
            else:
                inputs = Variable(batch['X'])
                labels = Variable(batch['Y'])

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Statistics 
            bs = inputs.size(0)
            loss = loss.item()
            loss_running.update(loss, bs)

            # measure elapsed time
            batch_time.update(time()-end)
            end = time()

            progress.display(iter)

            pred = outputs.data.cpu().numpy()
            N, _, h, w = pred.shape
            pred = pred.transpose(0, 2, 3, 1).reshape(-1, n_class).argmax(axis=1).reshape(N, h, w)

            target = batch['l'].cpu().numpy().reshape(N, h, w)
            for p, t in zip(pred, target):
                acc_running.update(pixel_acc(p, t))

            miou_running.update(iou(outputs, labels, n_class, ignore_class=0))
            acc2_running.update(pixel_acc2(outputs, labels, ignore_class=0))
            
            del batch
            torch.cuda.empty_cache()

    val_loss = loss_running.avg
    val_miou = miou_running.avg
    val_acc = acc_running.avg
    val_acc2 = acc2_running.avg
    print('\nvalidation loss {:.4f} | validation miou {:.4f} | validation acc {:.4f} | validation acc2 {:.4f}'.format(val_loss, val_miou, val_acc, val_acc2))

    return val_loss, val_miou, val_acc, val_acc2
