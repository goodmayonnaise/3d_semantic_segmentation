
import os, cv2, yaml
import numpy as np 
from glob import glob 
from einops import rearrange

from data_loader.laserscan import LaserScan
# from laserscan import LaserScan

import torch
from torch.utils.data import Dataset


class SemanticKITTI(Dataset):
    def __init__(self, data_path, shape, nclasses, mode, front, **kwargs):
        CFG = self.load_config()
        self.swap_dict = CFG['learning_map']
        sequences = CFG['split'][mode]
        self.sequences = [str(i).zfill(2) for i in sequences]
        self.path = self.data_path_load(self.sequences, data_path)
        self.opendata = LaserScan(front=front, project=True, sem_color_dict=CFG['color_map'])

        self.mode = mode
        self.nclasses = nclasses
        self.input_shape = shape     
        
        self.pcd_paths = self.path[0]
        self.label_paths = self.path[1]
        self.img_paths = self.path[2]

    def load_config(self):
        cfg_path = '/vit-adapter-kitti/jyjeon/data_loader/semantic-kitti.yaml'
        try:
            print("Opening config file %s" % cfg_path)
            CFG = yaml.safe_load(open(cfg_path, 'r'))
        except Exception as e:
            print(e)
            print("Error opening yaml file.")
            quit()
        return CFG

    def replace_with_dict(self, ar):
        # Extract out keys and values
        k = np.array(list(self.swap_dict.keys()))
        v = np.array(list(self.swap_dict.values()))

        # Get argsort indices
        sidx = k.argsort()
        
        # Drop the magic bomb with searchsorted to get the corresponding
        # places for a in keys (using sorter since a is not necessarily sorted).
        # Then trace it back to original order with indexing into sidx
        # Finally index into values for desired output.
        return v[sidx[np.searchsorted(k,ar,sorter=sidx)]]     

    def __len__(self):
        return len(self.pcd_paths)

    def __getitem__(self, idx):
        x, y, img = self.pcd_paths[idx], self.label_paths[idx], self.img_paths[idx]
        x, y = self.opendata.set_data(x, y)

        rem, depth, mask = x['remission'], x['range'], x['mask']
        rem = torch.FloatTensor(rem)
        rem = torch.unsqueeze(rem, 0)

        pad_rem = torch.FloatTensor(x['pad_remission'])
        pad_rem = torch.unsqueeze(pad_rem,0)
        # rdm = np.stack((rem, depth, mask), axis=0) 
        # rdm = torch.FloatTensor(rdm)

        # for img
        img = cv2.imread(img)
        img = cv2.resize(img, (self.input_shape[1], self.input_shape[0]))
        img = rearrange(img, 'h w c -> c h w')
        img = torch.FloatTensor(img)

        y = torch.FloatTensor(y['label'])
        h, w = y.shape 
        y_class = torch.zeros(self.nclasses, h, w)
        for c in range(self.nclasses):
            y_class[c] = (y==c).type(torch.int32).clone().detach()

        return {'img': img, 'label':y_class, 'rgb_label':y, 'rem':rem, 'pad_rem':pad_rem}

    def data_path_load(self, sequences, data_path): 
        img_paths = [os.path.join(data_path, sequence_num, "image_2") for sequence_num in sequences]
        pcd_paths = [os.path.join(data_path, sequence_num, "velodyne") for sequence_num in sequences]
        label_paths = [os.path.join(data_path, sequence_num, "labels") for sequence_num in sequences]
        
        pcd_names, label_names, img_names = [], [], []

        for pcd_path, label_path, img_path in zip(pcd_paths, label_paths, img_paths):    
            pcd_names = pcd_names + glob(str(os.path.join(os.path.expanduser(pcd_path),"*.bin")))
            label_names = label_names + glob(str(os.path.join(os.path.expanduser(label_path),"*.label")))
            img_names = img_names + glob(str(os.path.join(os.path.expanduser(img_path),"*.png")))

        pcd_names.sort()
        label_names.sort()
        img_names.sort()

        return pcd_names, label_names, img_names


if __name__ == "__main__":

    def load_config():
        cfg_path = '/vit-adapter-kitti/jyjeon/data_loader/semantic-kitti.yaml'
        try:
            print("Opening config file %s" % "config/semantic-kitti.yaml")
            import yaml
            CFG = yaml.safe_load(open(cfg_path, 'r'))
        except Exception as e:
            print(e)
            print("Error opening yaml file.")
            quit()
        return CFG

    data_path = '/vit-adapter-kitti/data/semantic_kitti/kitti/dataset/sequences'
    shape = (256, 1024)
    mode = "train"
    nclasses = 20 
    batch_size = 1
    dataset = SemanticKITTI(data_path, shape, nclasses, mode)
    from laserscan import LaserScan
    from torch.utils.data import DataLoader
    import cv2, torch 
    loader = DataLoader(dataset, batch_size, num_workers=1)
    
    for iter, batch in enumerate(loader):
        
        print()
        
    
