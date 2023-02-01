import torch 
from einops import rearrange

def _take_channels(*xs, ignore_channels=None): # xs[0] : outputs xs[1] : labels 
    if ignore_channels is None:
        return xs
    else:
        # ignore_channels를 제외하고 인덱스 다시 정리 
        channels = [channel for channel in range(xs[0].shape[1]) if channel != ignore_channels]
        xs = [torch.index_select(x, dim=1, index=torch.tensor(channels).to(x.device)) for x in xs]
        return xs

def get_confusion_matrix(pr, gt, num_classes, ignore_index):

    mask = (gt != ignore_index)
    pred_label = pr[mask]
    label = gt[mask]

    n = num_classes
    inds = n * label + pred_label

    mat = torch.bincount(inds, minlength=n**2).reshape(num_classes, num_classes)
    return mat

def IntersectionOverUnion(pr, gt, ignore_index=0, num_class = 19):
    pr, gt = _take_channels(pr, gt, ignore_channels=ignore_index)
    num_img = len(pr)
    total_matix = torch.zeros((num_class, num_class), dtype=torch.float).cuda()

    for i in range(num_img):
        pred = torch.argmax(pr[i], dim=0, keepdim=True)
        true = torch.argmax(gt[i], dim=0, keepdim=True)

        mat = get_confusion_matrix(pred, true, num_classes=num_class, ignore_index=ignore_index)     
        total_matix += mat.cuda()
    iou = torch.diagonal(total_matix) / (total_matix.sum(axis=1) + total_matix.sum(axis=0) - torch.diagonal(total_matix))
    return torch.nansum(iou)/num_img

# ----------------------------------------------------------------------------------------------------------------------------
def comput_confusion_matrix(pred, true, num_cls=19):
    true = rearrange(true, 'b c h -> b h c')
    pred = rearrange(pred, 'b c h -> b h c')
    conf_mat = torch.zeros((true.shape[0], num_cls, num_cls)) # 클래스 19개의 confusion matrix shape 초기 설정, 배치의 각 샘플마다 confusion matrix를 하나씩 쓰도록 함

    for i in range(true.shape[0]): # 각 배치마다 돌면서         
        y_args = torch.argmax(torch.Tensor(true[i][true[i].sum(axis=-1) > 0]), axis=-1)
        y_hat_args = torch.argmax(torch.Tensor(pred[i][true[i].sum(axis=-1) >0]), axis=-1) # boolean indexing을 이용해서 정답이 존재하는 포인트만 남긴 후 argmax를 통해 예측 값을 얻음
        
        inds = num_cls * y_args + y_hat_args
        conf_mat[i] = torch.bincount(inds, minlength=num_cls**2).reshape(num_cls, num_cls)  
        
    return conf_mat

def iou(pred, target, num_cls, ignore_class=None):
    if ignore_class == 0:  
        pred = pred[:,1:,:]
        target = target[:,1:,:]
    elif ignore_class == 19:
        pred = pred[:,:-1,:]
        target = target[:,:-1,:]
    
    target = rearrange(target, 'b c h w -> b c (h w)')
    pred = rearrange(pred, 'b c h w -> b c (h w)')

    conf_mat = comput_confusion_matrix(pred, target, num_cls=num_cls)
    miou = torch.zeros((conf_mat.shape[0]))
    for i in range(conf_mat.shape[0]):
        sum_over_row = torch.sum(conf_mat[i], axis=0)
        sum_over_col = torch.sum(conf_mat[i], axis=1)
        true_positives = conf_mat[i].diagonal()
        denominator = sum_over_row + sum_over_col - true_positives
        denominator = torch.where(denominator == 0, torch.nan, denominator) #???
        iou_each_class = true_positives / denominator
        miou[i] = torch.nansum(iou_each_class) / torch.sum(iou_each_class > 0)
    return torch.nanmean(miou)

# ----------------------------------------------------------------------------------------------------------------------------

def pixel_acc(pred, target, ignore_class=None):
    if ignore_class == 0:  
        pred = pred[:,1:,:]
        target = target[:,1:,:]
    elif ignore_class == 19:
        pred = pred[:,:-1,:]
        target = target[:,:-1,:]
    correct = (pred == target).sum()
    total   = (target == target).sum()
    return correct / total

def pixel_acc2(pred, target, ignore_class=None):
    if ignore_class == 0:  
        pred = pred[:,1:,:]
        target = target[:,1:,:]
    elif ignore_class == 19:
        pred = pred[:,:-1,:]
        target = target[:,:-1,:]

    target = rearrange(target, 'b c h w -> b (h w) c')
    pred = rearrange(pred, 'b c h w -> b (h w) c ')

    result = torch.zeros((target.shape[0],))  
    
    for i in range(target.shape[0]):
        y_args = torch.argmax(torch.Tensor(target[i][target[i].sum(axis=-1) > 0]), axis=-1)
        y_hat_args = torch.argmax(torch.Tensor(pred[i][target[i].sum(axis=-1) >0]), axis=-1)

        correct = (y_hat_args[i] == y_args[i]).sum()
        total   = (y_args[i] == y_args[i]).sum()
        result[i] = correct / total
    return torch.nanmean(result)

# # borrow functions and modify it from https://github.com/Kaixhin/FCN-semantic-segmentation/blob/master/main.py
# # Calculates class intersections over unions
# def iou(pred, target, n_class):
#     ious = []
#     for cls in range(n_class):
#         pred_inds = pred == cls
#         target_inds = target == cls
#         intersection = pred_inds[target_inds].sum()
#         union = pred_inds.sum() + target_inds.sum() - intersection
#         if union == 0:
#             ious.append(float('nan'))  # if there is no ground truth, do not include in evaluation
#         else:
#             ious.append(float(intersection) / max(union, 1))
#         # print("cls", cls, pred_inds.sum(), target_inds.sum(), intersection, float(intersection) / max(union, 1))
#     return ious


