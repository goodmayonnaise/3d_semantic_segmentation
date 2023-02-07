import os, yaml
import torch

def load_weight(ckpt_dir, model, device):
    dict_model = torch.load(ckpt_dir, map_location=device)
    model.load_state_dict(dict_model, strict=False)
    return model 

# 다음 학습 시 쓸 것 - test 필요 
# def load_weight(ckpt_dir, model, optimizer, epoch, device):
#     dict_model = torch.load(ckpt_dir, map_location=device)

#     model.load_state_dict(dict_model['model_state_dict'], strict=False)
#     optimizer.load_state_dict(dict_model['optimizer_state_dict'])
#     # epoch.load_state_dict(dict_model['epoch'])
#     return model, optimizer, epoch

def set_device(gpus):
    device = "cuda" if torch.cuda.is_available() else 'cpu'
    
    torch.manual_seed(777) # 정확한 테스트를 위한 random seed 고정 
    if device == 'cuda':
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"]= gpus 
        
        torch.cuda.manual_seed_all(777)
    return device

class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

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
