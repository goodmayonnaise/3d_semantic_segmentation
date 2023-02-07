
import os 
from time import time
from datetime import datetime
from torchsummaryX import summary as Xsummary
from torchinfo import summary as torchsummary

from utils import set_device, load_weight
from losses import FocalLosswithDiceRegularizer
from model import KSC2022, KSC2022_Fusion
from DataLoaders.cityscapes import CityScapesDataset
from DataLoaders.KITTI import load_data
from DataLoaders.SemanticKITTI import load_semanticKITTI
from pytorchtools import EarlyStopping
from train import train
from test import _test

import torch 
import torch.nn as nn
from torch.optim import lr_scheduler, Adam
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


if __name__ == "__main__":
    # -------------------------------------------------------------------------- setting parameter ---------------------------------------------------------------------------
    fusion_lev = "none" # none, early, mid_stage1~4, late 

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
    gpus = os.environ["CUDA_VISIBLE_DEVICES"]
    num_workers = len(gpus.split(",")) * 2

    phase = "train" # train /transfer_learning / test 
    freeze_cnt = 150
    dataset = "cityscapes" # cityscapes / kitti / city_kitti / semantic_kitti
    n_class    = 20
    input_shape = (384//4, 1280//4) # 96 312
    criterion = FocalLosswithDiceRegularizer(reduction="mean") # setting loss 
      
    epochs     = 500
    lr         = 1e-4
    momentum   = 0
    w_decay    = 1e-5
    step_size  = 50
    gamma      = 0.5

    cityscapes_path = "/mnt/team_gh/cityscapes"
    kitti_path = "/mnt/data/home/team_gh/data/KITTI_Semantics_aug"
    semantickitti_path = '/mnt/data/home/team_gh/data/kitti/dataset/f_sequences'
    
    device = set_device(gpus) 
    use_gpu = torch.cuda.is_available()
    num_gpu = list(range(torch.cuda.device_count()))

    if phase in ["train", "transfer_learning"]:
        if dataset in ["cityscapes", "city_kitti"]:
            if dataset == "cityscapes":    
                batch_size = 50                  # <= 5 
                train_file = os.path.join(cityscapes_path, "train_aug.csv") # train.csv / train_aug.csv 
                val_file   = os.path.join(cityscapes_path, "val.csv") 
            else : # city_kitti
                batch_size = 28                  # <= 5 
                train_file = os.path.join(cityscapes_path, "city_kitti_train_aug.csv") 
                val_file   = os.path.join(cityscapes_path, "city_kitti_val.csv")
            train_data = CityScapesDataset(csv_file=train_file, input_shape=input_shape, n_class=n_class)
            train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)
            val_data = CityScapesDataset(csv_file=val_file, input_shape=input_shape, n_class=n_class)
            val_loader = DataLoader(val_data, batch_size=batch_size, num_workers=num_workers)

        elif dataset == "kitti":
            batch_size = 40                  # <= 5 
            train_loader, val_loader = load_data(batch_size=batch_size, phase="train", input_shape=input_shape, num_workers=num_workers, data_path=kitti_path)

        elif dataset == "semantic_kitti":
            batch_size = 40 
            train_loader, val_loader = load_semanticKITTI(batch_size=batch_size, phase=phase, data_path=semantickitti_path, num_workers=num_workers, input_shape=input_shape)

        configs = "{}_batch{}_epoch{}_Adam_scheduler-step{}-gamma{}_lr{}_momentum{}_w_decay{}".format(dataset, batch_size, epochs, step_size, gamma, lr, momentum, w_decay)
        print("Configs:", configs)

        # create dir for model
        model_dir = "weights"
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        model_path = os.path.join(model_dir, configs)

        if dataset == "semantic_kitti":
            model = KSC2022_Fusion(input_shape=input_shape, fusion_lev=fusion_lev)
        else :
            model = KSC2022(input_shape=input_shape, fusion_lev=fusion_lev, n_class=n_class)

        if phase == "transfer_learning":
            model = load_weight(ckpt_dir='./weights/Cityscapes_aug_best_checkpoint.pt', model=model, device=device)
            freeze = 0 
            for param in model.parameters():
                if freeze < freeze_cnt:
                    param.requires_grad = False 
                freeze += 1 
                
        # print(Xsummary(model, torch.rand((5, 3, 384//4, 1280//4))))
        print(torchsummary(model, (5, 3, 384//4, 1280//4)))    
        
        if use_gpu:
            ts = time()
            model = model.cuda()
            model = nn.DataParallel(model, device_ids=num_gpu)
            print("Finish cuda loading, time elapsed {}".format(time() - ts))
        
        optimizer = Adam(model.to(device).parameters(), lr=lr)
        # scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)  # decay LR by a factor of 0.5 every 30 epochs
        # scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=0.001)
        # scheduler = lr_scheduler.CyclicLR(optimizer, base_lr=0.001, max_lr=0.1, step_size_up=50, step_size_down=None, mode='exp_range', gamma=0.995, cycle_momentum=False)
        
        board_path = "log"
        if not os.path.exists(board_path):
            os.makedirs(board_path)
        metrics = {'train_loss':[], 'train_miou':[], 'train_acc':[], 'train_acc2':[],
                   'val_loss':[], 'val_miou':[], 'val_acc':[], 'val_acc2':[]}
        
        writer_train = SummaryWriter(log_dir=os.path.join(board_path,'train'))
        writer_val = SummaryWriter(log_dir = os.path.join(board_path,'val'))

        # Generate log file                 --------------------------------------------------
        with open('./log/result.csv', 'a') as epoch_log:
            epoch_log.write('\n--------------new--------------\nepoch\ttrain loss\tval loss\ttrain miou\tval miou\ttrain acc\tval acc\t\ttrain acc2\tval acc2\n')

        early_stopping = EarlyStopping(patience=10, verbose=True, path='./weights/earlystop_weights.pt')
        t_s = datetime.now()
        print(f'\ntrain start time : {t_s}')
        train(epochs, train_loader, val_loader, model, optimizer, use_gpu, criterion, dataset,
              n_class, metrics, scheduler, writer_train, writer_val, early_stopping, device)
        print(f'\ntrain start time : {t_s}, end of train : {datetime.now()}')


    elif phase == "test":
        if dataset in ["cityscapes", "city_kitti"]:
            cityscapes_path = "/mnt/data/home/team_gh/data/cityscapes"
            if dataset == "cityscapes":
                batch_size = 50 # 수정 예정
                test_file = os.path.join(cityscapes_path, "val.csv")
            elif dataset == "city_kitti":
                batch_size = 20 # 수정 예정
                test_file = os.path.join(cityscapes_path, "city_kitti_test.csv")
            test_data = CityScapesDataset(csv_file=test_file, phase="test")
            test_loader = DataLoader(test_data, batch_size=batch_size)

        elif dataset == "kitti":
            batch_size = 40 # 수정 예정
            kitti_paths = '/mnt/data/home/team_gh/data/KITTI_Semantics'
            test_loader = load_data(batch_size=batch_size, phase="test", input_shape=input_shape, num_workers=num_workers, data_path=kitti_paths)
            
        model = KSC2022(input_shape=input_shape, fusion_lev=fusion_lev, n_class=n_class)
        
        if use_gpu:
            ts = time()
            model = model.cuda()
            model = nn.DataParallel(model, device_ids=num_gpu)
            print("Finish cuda loading, time elapsed {}".format(time() - ts))
        
        optimizer = Adam(model.to(device).parameters(), lr=lr)
        
        ckpt_dir = "./weights"
        for ckpt in os.listdir(ckpt_dir):
            net = load_weight(ckpt_dir=os.path.join(ckpt_dir,ckpt), model=model, device=device)
            print(f'\n-------------------------------------------------')
            print(f'start ckpt name\t : {os.path.join(ckpt_dir,ckpt)}\ndataset\t\t : {dataset}')

            t_s = datetime.now()
            print(f'\ntest start time : {t_s}')
            _test(test_loader, net, use_gpu, criterion, n_class, device)        
            print(f'\ntest start time : {t_s}, end of test : {datetime.now()}')   


    