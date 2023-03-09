
import cv2, pickle, yaml, os
import numpy as np 
from tqdm import tqdm
from einops import rearrange

from model import new_model
from utils import set_device


import torch
from torch import nn
from torch.optim import lr_scheduler, Adam
import torchvision.transforms as T


def point_padding(wh_list, y_img, H, W, fill_size = 1):        
    for h, w in zip(wh_list[0], wh_list[1]):
        if fill_size<=h and h <=H-fill_size-1 and fill_size<=w and w<=W-fill_size-1:
            y_img[:,h,w+fill_size] = y_img[:,h,w]
            y_img[:,h,w-fill_size] = y_img[:,h,w]
            y_img[:,h+fill_size,w] = y_img[:,h,w]
            y_img[:,h-fill_size,w] = y_img[:,h,w]
            y_img[:,h+fill_size,w+fill_size] = y_img[:,h,w]
            y_img[:,h-fill_size,w-fill_size] = y_img[:,h,w]
            y_img[:,h+fill_size,w-fill_size] = y_img[:,h,w]
            y_img[:,h-fill_size,w+fill_size] = y_img[:,h,w]
        elif fill_size<=h and h <=H-fill_size-1 and fill_size>w:
            y_img[:,h,w+fill_size] = y_img[:,h,w]
            y_img[:,h+fill_size,w] = y_img[:,h,w]
            y_img[:,h-fill_size,w] = y_img[:,h,w]
            y_img[:,h+fill_size,w+fill_size] = y_img[:,h,w]
            y_img[:,h-fill_size,w+fill_size] = y_img[:,h,w]
        elif fill_size<=h and h <=H-fill_size-1 and w>W-fill_size-1:
            y_img[:,h,w-fill_size] = y_img[:,h,w]
            y_img[:,h+fill_size,w] = y_img[:,h,w]
            y_img[:,h-fill_size,w] = y_img[:,h,w]
            y_img[:,h-fill_size,w-fill_size] = y_img[:,h,w]
            y_img[:,h+fill_size,w-fill_size] = y_img[:,h,w]
        elif fill_size>h and fill_size<=w and w<=W-fill_size-1:
            y_img[:,h,w+fill_size] = y_img[:,h,w]
            y_img[:,h,w-fill_size] = y_img[:,h,w]
            y_img[:,h+fill_size,w] = y_img[:,h,w]
            y_img[:,h+fill_size,w+fill_size] = y_img[:,h,w]
            y_img[:,h+fill_size,w-fill_size] = y_img[:,h,w]
        elif h > H-fill_size-1 and fill_size<=w and w<=W-fill_size-1:
            y_img[:,h,w+fill_size] = y_img[:,h,w]
            y_img[:,h,w-fill_size] = y_img[:,h,w]
            y_img[:,h-fill_size,w] = y_img[:,h,w]
            y_img[:,h-fill_size,w-fill_size] = y_img[:,h,w]
            y_img[:,h-fill_size,w+fill_size] = y_img[:,h,w]
        elif fill_size>h and fill_size>w :
            y_img[:,h,w+fill_size] = y_img[:,h,w]
            y_img[:,h+fill_size,w] = y_img[:,h,w]
            y_img[:,h+fill_size,w+fill_size] = y_img[:,h,w]
        elif fill_size>h and w>W-fill_size-1:
            y_img[:,h,w-fill_size] = y_img[:,h,w]
            y_img[:,h+fill_size,w] = y_img[:,h,w]
            y_img[:,h+fill_size,w-fill_size] = y_img[:,h,w]
        elif h > H-fill_size-1 and fill_size>w :
            y_img[:,h,w+fill_size] = y_img[:,h,w]
            y_img[:,h-fill_size,w] = y_img[:,h,w]
            y_img[:,h-fill_size,w+fill_size] = y_img[:,h,w]
        elif h>H-fill_size-1 and w>W-fill_size-1:
            y_img[:,h,w-fill_size] = y_img[:,h,w]
            y_img[:,h-fill_size,w] = y_img[:,h,w]
            y_img[:,h-fill_size,w-fill_size] = y_img[:,h,w]
    return y_img


def colorize_sample(d_arg, point_2D_from_3D_x, point_2D_from_3D_y, label, CFG):# d_arg는 swap dict가 끝난 클래스
        color_dict = CFG["color_map"] # config내 color map bgr
        max_sem_key = 0
        for key, data in color_dict.items(): # max_sem_key 설정 , color_dict의 key에는 swap이 안된 클래스의 색상 data
            if key + 1 > max_sem_key:
                max_sem_key = key + 1 # 260개
        sem_color_lut = np.zeros((max_sem_key + 100, 3), dtype=np.float32) 
        for key, value in color_dict.items():
            sem_color_lut[key] = np.array(value, np.float32) 
        # print(sem_color_lut[0:20])
        d_color_origin = np.zeros(shape=(d_arg.shape[0],d_arg.shape[1],3))
        for x,y,l in zip(point_2D_from_3D_x, point_2D_from_3D_y, label):
            d_color_origin[y][x]=sem_color_lut[l]

        d_color=np.zeros(shape=(d_arg.shape[0],d_arg.shape[1],3))
        for i in range(len(d_arg)):
            for j in range(len(d_arg[i])):
                d_color[i][j] = sem_color_lut[CFG["learning_map_inv"][d_arg[i][j]]]
                
        return d_color, d_color_origin # pred, uv_laebel 


def replace_with_dict(ar, learning_map):
    # Extract out keys and values
    k = np.array(list(learning_map.keys()))
    v = np.array(list(learning_map.values()))

    # Get argsort indices
    sidx = k.argsort()
    
    # Drop the magic bomb with searchsorted to get the corresponding
    # places for a in keys (using sorter since a is not necessarily sorted).
    # Then trace it back to original order with indexing into sidx
    # Finally index into values for desired output.
    return v[sidx[np.searchsorted(k,ar,sorter=sidx)]]    

def load_config():
    cfg_path = './config/semantic-kitti.yaml'
    try:
        print("Opening config file %s" % "config/semantic-kitti.yaml")
        CFG = yaml.safe_load(open(cfg_path, 'r'))
    except Exception as e:
        print(e)
        print("Error opening yaml file.")
        quit()
    return CFG 


if __name__ == "__main__":
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
    gpus = os.environ["CUDA_VISIBLE_DEVICES"]
    num_workers = len(gpus.split(",")) * 2
    device = set_device(gpus) 
    use_gpu = torch.cuda.is_available()
    num_gpu = list(range(torch.cuda.device_count()))
    
    input_shape = (256, 1280)
    
    CFG = load_config()
    learning_map = CFG['learning_map']
    learning_map_inv = CFG['learning_map_inv']
    color_map = CFG['color_map']
    
    transform = T.ToPILImage()

    sequence_n = '00'
    img_dir = f'/mnt/team_gh/kitti/dataset/sequences/{sequence_n}/image_2'
    img_files = os.listdir(img_dir)
    img_filenames = []
    for img_file in img_files:
        img_filenames.append(img_file[:-4])    
    
    for file_n in tqdm(img_filenames):
        x_img = f'/mnt/team_gh/kitti/dataset/sequences/{sequence_n}/image_2/{file_n}.png'
        x_rem = f'/mnt/team_gh/kitti/dataset/sequences/{sequence_n}/projection_front/remission/{file_n}.npy'
        mapping_data = f'/mnt/team_gh/kitti/dataset/sequences/{sequence_n}/proj_point_label_mapping/{file_n}.pickle' # output y_img, y_rem 
        
        x_rem = np.load(x_rem)
        x_rem = np.array(x_rem)
        min_H = 0
        for i in x_rem:
            if sum(i!=-1):
                break
            min_H += 1
        
        H, W = input_shape[0], input_shape[1]
        
        x_img = cv2.imread(x_img)
        x_img = np.array(x_img)
        x_img = cv2.resize(x_img[min_H:,:], (W, H))
        cv2.imwrite(f'./samples/rgb/{sequence_n}_{file_n}_rgb.png', x_img)
        x_img = torch.FloatTensor(x_img)
        x_img = rearrange(x_img, 'h w c -> c h w') # 20 256 1280
        
        x_rem_H, x_rem_W = x_rem.shape # 256 1280 
        
        x_rem = cv2.resize(x_rem[min_H:,:], (W,H))
        x_rem = np.expand_dims(x_rem, axis=-1)
        x_rem = torch.FloatTensor(x_rem)
        x_rem = rearrange(x_rem, 'h w c -> c h w') # 20 256 1280
        
        with open(mapping_data, 'rb') as f:
            y_map = pickle.load(f)
        
        v, u = y_map['uv']  # 19369
        u, v = u*(W/x_rem_W), (v-min_H)*(H/(x_rem_H-min_H))
        u, v = np.round(u,0).astype(np.int32), np.round(v,0).astype(np.int32)
        
        uv_label = np.array(y_map['label']) # n = 19369
        
        y_rem = np.zeros(shape=(20, H, W))
        y_rem[replace_with_dict(uv_label, learning_map), v, u] = 1
        
        y_img = y_rem
        wh_list = np.where(np.sum(y_rem, axis=0)!=0)
        y_img = point_padding(wh_list, y_img, H, W, 2)
        y_img[0, np.sum(y_img, axis=0)==0] = 1 # numpy 에서 cv2.imwrite
        # y_color, y_color_origin = colorize_sample(np.argmax(y_img,axis=0), u, v, uv_label, CFG)  
        # cv2.imwrite("./samples/label_fill2.png", y_color)
        # cv2.imwrite('./samples/label_origin.png',y_color_origin)
        y_img = torch.FloatTensor(y_img)
        
        y_rem[0, np.sum(y_rem, axis=0)==0] = 1
        y_rem = rearrange(y_rem, 'c h w -> c (h w)')
        y_rem = torch.FloatTensor(y_rem)
        
        x_img, x_rem, y_img, y_rem = x_img.unsqueeze(0), x_rem.unsqueeze(0), y_img.unsqueeze(0), y_rem.unsqueeze(0)
        # x_img, x_rem, y_img, y_rem = x_img.to(device).unsqueeze(0), x_rem.to(device).unsqueeze(0), y_img.to(device).unsqueeze(0), y_rem.to(device).unsqueeze(0)
        
        # state = torch.load('./weights/best_weight/random_states_3.pkl')
        state = torch.load('./weights/best_weight/pytorch_model.bin')
        model = new_model(input_shape=input_shape)
        # model = model.cuda()
        # model = nn.DataParallel(model, device_ids=num_gpu)
        # model.load_state_dict(state['model_state_dict'])
        model.load_state_dict(state)
        
        optimizer = Adam(model.parameters(), lr=1e-4)
        schdeuler = lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
        
        with torch.no_grad():
            model.eval()
            segment_out, uv_out, uv = model(x_img, x_rem) 
            segment_out, uv_out , uv = segment_out.squeeze(0), uv_out.squeeze(0), uv.squeeze(0) # 20 256 1280 / 20 256*1280
            '''
            segment out     20 256 1280
            uv out          20 256*1280
            uv              32 256 1280 
            '''
            segment_arr = segment_out.cpu().numpy()
            segment_arr = np.argmax(segment_arr, axis=0)
            
            segment_result = np.zeros([*segment_arr.shape, 3])
            
            for c in color_map.keys():
                segment_result[segment_arr==c] = color_map[c][::-1]
            
            cv2.imwrite(f"./samples/rgb_out/{sequence_n}_{file_n}_rgb_out.png", segment_result)

            uv_arr = uv_out.cpu().numpy()
            uv_arr = rearrange(uv_arr, 'c (h w) -> c h w',  h=256) # 20 256 1280
            
            uv_arr = np.argmax(uv_arr, axis=0) #  1 256 1280
            
            # print(uv_arr.max()) # 19 
            # print(uv_arr.min()) # 1 
            
            result = np.zeros([H, W])
            
            for x, y in zip(u, v):
                result[y, x] = uv_arr[y, x]
            
            uv_pred, uv_label = colorize_sample(result, u, v, uv_label, CFG)
            
            cv2.imwrite(f"./samples/uv_pred/{sequence_n}_{file_n}_pred.png", uv_pred)
            cv2.imwrite(f'./samples/uv_label/{sequence_n}_{file_n}_uv_label.png', uv_label)
        
