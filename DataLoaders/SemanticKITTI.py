
import os, cv2 
import numpy as np 
from glob import glob 
from einops import rearrange
import torch

from utils import load_config

from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn.functional import one_hot


class SemnaticKITTI(Dataset):
    def __init__(self, input_file_pathes, label_file_pathes, input_shape, swap_dict, num_cls, train_phase=True, **kwargs):
        # super(Dataset,self).__init__(**kwargs)
        
        self.image2_pathes = input_file_pathes[0]
        self.remission_pathes = input_file_pathes[-1]
        self.depth_pathes = label_file_pathes[0]
        self.label_pathes = label_file_pathes[-1]
        self.swap_dict = swap_dict
        self.num_cls = num_cls
        self.input_shape = input_shape     
        self.train_phase = train_phase
        # self.on_epoch_end()    

    def replace_with_dict(self, ar, dic):
        # Extract out keys and values
        k = np.array(list(dic.keys()))
        v = np.array(list(dic.values()))

        # Get argsort indices
        sidx = k.argsort()
        
        # Drop the magic bomb with searchsorted to get the corresponding
        # places for a in keys (using sorter since a is not necessarily sorted).
        # Then trace it back to original order with indexing into sidx
        # Finally index into values for desired output.
        return v[sidx[np.searchsorted(k,ar,sorter=sidx)]]     

    def __len__(self):
        return len(self.image2_pathes)

    def __getitem__(self, idx):

        x_img = self.image2_pathes[idx]
        x_img = cv2.imread(x_img)
        x_img = np.array(x_img)
        x_img = cv2.resize(x_img, (self.input_shape[1], self.input_shape[0]))
        x_img = torch.FloatTensor(x_img)
        x_img = rearrange(x_img, 'h w c -> c h w')

        x_rem = self.remission_pathes[idx]
        x_rem = np.load(x_rem)
        x_rem = np.array(x_rem)
        x_rem = cv2.resize(x_rem, (self.input_shape[1], self.input_shape[0]))
        x_rem = np.expand_dims(x_rem, axis=-1)
        x_rem = torch.FloatTensor(x_rem)
        x_rem = rearrange(x_rem, 'h w c -> c h w')

        y_img = self.label_pathes[idx]
        y_img = np.load(y_img)
        y_img = self.replace_with_dict(y_img, self.swap_dict)
        y_img = cv2.resize(y_img, (self.input_shape[1], self.input_shape[0]), interpolation=cv2.INTER_NEAREST)
        y_img = torch.from_numpy(y_img).float()
        y_img = one_hot(y_img.to(torch.int64), num_classes=self.num_cls)
        y_img = np.array(y_img)
        y_img = torch.FloatTensor(y_img)
        y_img = rearrange(y_img, 'h w c -> c h w')

        y_rem = self.depth_pathes[idx]
        y_rem = np.load(y_rem)
        y_rem = cv2.resize(y_rem, (self.input_shape[1], self.input_shape[0]))
        y_rem = np.expand_dims(y_rem, axis=-1)
        y_rem = torch.FloatTensor(y_rem)
        y_rem = rearrange(y_rem, 'h w c -> c h w')

        return {'X':x_img, 'X_rem':x_rem, 'Y':y_img, 'Y_rem':y_rem}


def data_path_load(sequences, data_path): 

    image2_paths = [os.path.join(data_path, sequence_num, "image_2") for sequence_num in sequences]
    remission_paths = [os.path.join(data_path, sequence_num, "projection_front", "remission") for sequence_num in sequences]
    depth_paths = [os.path.join(data_path, sequence_num, "projection_front", "depth") for sequence_num in sequences]
    sem_label_paths = [os.path.join(data_path, sequence_num, "label_projection_front", "sem_label") for sequence_num in sequences]
    
    image2_names = list()
    remission_names = list()
    depth_names = list()
    sem_label_names = list()

    for image2_path, remission_path, depth_path, sem_label_path in zip(image2_paths, remission_paths, depth_paths, sem_label_paths):    
        image2_names = image2_names + glob(str(os.path.join(os.path.expanduser(image2_path),"*.png")))
        remission_names = remission_names + glob(str(os.path.join(os.path.expanduser(remission_path),"*.npy")))
        depth_names = depth_names + glob(str(os.path.join(os.path.expanduser(depth_path),"*.npy")))
        sem_label_names = sem_label_names + glob(str(os.path.join(os.path.expanduser(sem_label_path),"*.npy")))

    image2_names.sort()
    remission_names.sort()
    depth_names.sort()
    sem_label_names.sort()

    return image2_names, remission_names, depth_names, sem_label_names


def load_semanticKITTI(batch_size, phase, data_path, num_workers, input_shape):

    CFG = load_config()
    learning_map = CFG['learning_map']
    sequences = CFG['split'][phase]
    sequences = [str(i).zfill(2) for i in sequences]

    image2_paths, remission_paths, depth_paths, sem_label_paths = data_path_load(sequences, data_path=data_path)  

    dataset = SemnaticKITTI(input_file_pathes=[image2_paths,remission_paths], 
                            label_file_pathes=[depth_paths,sem_label_paths], 
                            input_shape=input_shape,
                            swap_dict=learning_map, 
                            num_cls=20, 
                            train_phase=True) # x_img, x_rem, y_img, y_rem

    dataset_size = len(dataset)
    train_size = int(dataset_size*0.8)
    val_size = dataset_size - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, num_workers=num_workers, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, num_workers=num_workers, batch_size=batch_size, shuffle=True)

    return train_loader, val_loader
