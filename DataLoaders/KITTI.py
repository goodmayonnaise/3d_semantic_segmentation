import numpy as np
import cv2, os 
from einops import rearrange

import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn.functional import one_hot


class KITTI(Dataset):
    def __init__(self, file_paths, input_shape, num_cls, train_phase=True, **kwargs):
        self.image2_pathes = file_paths[0]
        self.label_pathes = file_paths[1]
        self.num_cls = num_cls
        self.input_shape = input_shape     
        self.train_phase = train_phase

    def __len__(self):
        return len(self.image2_pathes)

    def __getitem__(self, idx):

        img = self.image2_pathes[idx]
        img = cv2.imread(img)
        img = np.array(img)
        img = cv2.resize(img, (self.input_shape[1], self.input_shape[0]))
        img = torch.FloatTensor(img)
        img = rearrange(img, 'h w c -> c h w')

        label = self.label_pathes[idx]
        label = cv2.imread(label)
        label = cv2.cvtColor(label, cv2.COLOR_BGR2GRAY)
        label = cv2.resize(label, (self.input_shape[1], self.input_shape[0]))
        label = np.array(label)
        label = torch.FloatTensor(label)

        h, w = label.size()
        target = torch.zeros(self.num_cls, h, w)
        for c in range(self.num_cls):
            target[c] = (label==c).type(torch.int32).clone().detach()

        # label = rearrange(label, 'h w c -> c h w')

        return {'X' : img, 'Y' : target, 'l' : label}


def data_path_load(data_path="/mnt/team_gh/KITTI_Semantics", phase="train"): 
    # if phase == "train":
    image2_x_paths = [os.path.join(data_path, "training", "image_2", filename) for filename in os.listdir(os.path.join(data_path, "training", "image_2"))]
    image2_y_paths = [os.path.join(data_path, "training", "semantic", filename) for filename in os.listdir(os.path.join(data_path, "training", "semantic"))]
    
    # elif phase == "test":
        
    #     image2_x_paths = [os.path.join(data_path, "image_2", filename) for filename in os.listdir(os.path.join(data_path, "image_2"))]
    #     image2_y_paths = [os.path.join(data_path, "semantic", filename) for filename in os.listdir(os.path.join(data_path, "semantic"))]
        

    image2_x_paths.sort()
    image2_y_paths.sort()

    return [image2_x_paths, image2_y_paths]


def load_data(batch_size, phase, input_shape, num_workers, data_path):

    data_paths = data_path_load(data_path=data_path, phase=phase)  

    if phase == "train":
        train_phase = True
    else:
        train_phase = False 
        
    dataset = KITTI(data_paths, 
                    input_shape=input_shape,# (384//4, 1280//4)
                    num_cls=20, 
                    train_phase=train_phase) 

    dataset_size = len(dataset)
    if phase == "train":
        train_size = int(dataset_size*0.8)
        val_size = dataset_size - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=train_phase, num_workers=num_workers)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=train_phase, num_workers=num_workers)

        return train_dataloader, val_dataloader
    elif phase == "test":
        test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=train_phase, num_workers=num_workers)
        return test_loader


# if __name__ == "__main__":
#     data_path = data_path_load()
#     train_dataloader, val_dataloader = load_data(batch_size=5, phase="train", input_shape=(1024//4, 2048//4))