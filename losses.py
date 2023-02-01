import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import CrossEntropyLoss
# from lovasz_softmax import lovasz_softmax

# class LovaszSoftmax(nn.Module):
#     def __init__(self, classes='present', per_image=False, ignore_index=0):
#         super(LovaszSoftmax, self).__init__()
#         self.smooth = classes
#         self.per_image = per_image
#         self.ignore_index = ignore_index
    
#     def forward(self, output, target):
#         logits = F.softmax(output, dim=1)
#         loss = lovasz_softmax(logits, target, ignore=self.ignore_index)
#         return loss

class DiceLoss(nn.Module):
    """Dice loss, need one hot encode input
    Args:
        weight: An array of shape [num_classes,]
        ignore_index: class index to ignore
        predict: A tensor of shape [N, C, *]
        target: A tensor of same shape with predict
        other args pass to BinaryDiceLoss
    Return:
        same as BinaryDiceLoss
    """
    def __init__(self, smooth=1, p=2, ignore_index=None, reduction='mean'):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.p = p
        self.ignore_index = ignore_index
        self.reduction=reduction

    def forward(self, predict, target):
        assert predict.shape == target.shape, 'predict & target shape do not match'

        predict = F.softmax(predict, dim=1)
        predict = predict.contiguous().view(predict.shape[0], -1)#torch.Size([8, 4456448])
        target = target.contiguous().view(target.shape[0], -1)

        num = 2 * (torch.sum(predict * target, dim=1) + self.smooth)
        den = torch.sum(predict ** self.p, dim=1) + torch.sum(target ** self.p, dim=1) + self.smooth

        loss = 1 - num / den
        # total loss는 배치 내 각 샘플의 손실값#torch.Size([8])
        if self.reduction == 'mean':
            loss = loss.mean()# 한 배치의 평균 손실값#torch.Size([])
        elif self.reduction == 'sum':
            loss = loss.sum()
        elif self.reduction == 'none':
            loss = loss

        return loss

class CategoricalCrossEntropyLoss(nn.Module):
    """Categorical Cross Entropy loss, need one hot encode input
    Args:
        weight: An array of shape [num_classes,]
        ignore_index: class index to ignore
        predict: A tensor of shape [N, C, *]
        target: A tensor of same shape with predict
        other args pass to BinaryFocalLoss
    Return:
        same as CrossEntropyLoss
    """
    def __init__(self, weight=None, ignore_index=None, reduction='mean', **kwargs):
        super(CategoricalCrossEntropyLoss, self).__init__()
        self.kwargs = kwargs
        self.weight = weight
        self.ignore_index = ignore_index
        self.reduction=reduction
        self.CEloss = CrossEntropyLoss(reduction=self.reduction,**self.kwargs)

    def forward(self, predict, target):
        assert predict.shape == target.shape, 'predict & target shape do not match'
        predict = F.softmax(predict, dim=1)  # prob

        if self.ignore_index==0:
            target = target[:,1:,:,:]# one-hot
            predict = predict[:,1:,:,:]# predicted prob.

        term_true = - torch.log(predict)
        term_false = - torch.log(1-predict)
        loss = torch.sum(term_true * target + term_false * (1-target), dim=1) #torch.Size([8, 256, 512])

        if self.reduction == "mean":# torch.Size([]) # loss: 4.507612  [    0/ 2975]
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()
        elif self.reduction == "none":
            pass
        
        # loss = self.CEloss(predict,torch.argmax(target, dim=1)) # torch.Size([]) # loss: 3.569603  [    0/ 2975]
        return loss

class FocalLoss(nn.Module):
    """Focal loss, need one hot encode input
    Args:
        weight: An array of shape [num_classes,]
        ignore_index: class index to ignore
        predict: A tensor of shape [N, C, *]
        target: A tensor of same shape with predict
        other args pass to BinaryFocalLoss
    Return:
        same as BinaryFocalLoss
    """
    def __init__(self,  alpha:float = 0.25, gamma:float = 2, eps = 1e-8, ignore_index=None, reduction:str ='mean'):
        super(FocalLoss, self).__init__()
        self.ignore_index = ignore_index        
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.eps = eps

    def forward(self, predict: Tensor, target: Tensor):
        assert predict.shape == target.shape, 'predict & target shape do not match'
        predict = F.softmax(predict, dim=1) + self.eps # prob
        
        if self.ignore_index==0:
            target = target[:,1:,:,:]
            predict = predict[:,1:,:,:]

        term_true =  - self.alpha * ((1 - predict) ** self.gamma) * torch.log(predict) # 틀리면 손실 커짐, 맞을수록 작아짐
        term_false = - (1-self.alpha) * (predict**self.gamma) * torch.log(1-predict) # 틀리면 손실 커짐, 맞을수록 작아짐
        loss = torch.sum(term_true * target + term_false * (1-target), dim=1)#torch.Size([8, 256, 512]) 
        # print(loss)
        # (x,y) 한 점의 클래스별 손실 값의 합
        # loss = torch.sum(loss, dim=(-2,-1))
        # 배치 내 각 샘플의 모든 지점의 손실 값#torch.Size([8])
        if self.reduction == "mean":
            loss = loss.mean()# 한 배치의 평균 손실값#torch.Size([]) # alpha가 0.25일때 loss : 0.8~ / alpha가 0.75일때 loss : 2.7~
        elif self.reduction == "sum":
            loss = loss.sum()
        elif self.reduction == "none":
            pass
        
        return loss

class FocalLosswithDiceRegularizer(nn.Module):
    """Focal loss with Dice loss as regularizer, need one hot encode input
    Args:
        weight: An array of shape [num_classes,]
        ignore_index: class index to ignore
        predict: A tensor of shape [N, C, *]
        target: A tensor of same shape with predict
        other args pass to BinaryFocalLoss
    Return:
        same as BinaryFocalLoss
    """
    def __init__(self,  alpha:float = 0.75, gamma:float = 2, eps = 1e-8, smooth=1, p=2,  ignore_index=None, reduction:str ='sum'):
        super(FocalLosswithDiceRegularizer, self).__init__()
        self.ignore_index = ignore_index        
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.eps = eps
        self.smooth = smooth
        self.p = p
        self.focal_loss = FocalLoss(alpha = self.alpha, gamma = self.gamma, eps = self.eps, ignore_index = self.ignore_index, reduction=reduction)
        self.dice_regularizer = DiceLoss(smooth=self.smooth, p=self.p, ignore_index = self.ignore_index, reduction=reduction)

    def forward(self, predict: Tensor, target: Tensor):
        assert predict.shape == target.shape, 'predict & target shape do not match'
        f_loss = self.focal_loss(predict, target)
        d_regularization = self.dice_regularizer(predict*target, target)
        # print(f_loss, d_regularization)#스케일의 차이가 너무 심하긴 한데 학습 진행 상황 체크 필요
        # tensor(0.7593, device='cuda:0', grad_fn=<MeanBackward0>) # alpha = 0.25
        # tensor(0.9197, device='cuda:0', grad_fn=<MeanBackward0>) # 
        return f_loss + (8 * d_regularization)