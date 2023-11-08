import torch
from torch import nn
from torch.nn import functional as F

class ASPP(nn.Module):
    def __init__(self, in_dim, out_dim, d_size):
        super(ASPP, self).__init__()

        self.aspp1 = nn.Sequential(nn.Conv2d(in_dim, out_dim, 1, 1),
                                   nn.BatchNorm2d(out_dim),
                                   nn.ReLU())
        self.aspp2 = nn.Sequential(nn.Conv2d(in_dim, out_dim, 3, 1, d_size, dilation=d_size),
                                   nn.BatchNorm2d(out_dim),
                                   nn.ReLU())
        self.aspp3 = nn.Sequential(nn.Conv2d(in_dim, out_dim, 3, 1, d_size*2, dilation=d_size*2),
                                   nn.BatchNorm2d(out_dim),
                                   nn.ReLU())
        self.aspp4 = nn.Sequential(nn.Conv2d(in_dim, out_dim, 3, 1, d_size*3, dilation=d_size*3),
                                   nn.BatchNorm2d(out_dim),
                                   nn.ReLU())
        self.aspp5 = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                   nn.Conv2d(in_dim, out_dim, 1, 1),
                                   nn.BatchNorm2d(out_dim),
                                   nn.ReLU())

        self.out = nn.Sequential(nn.Conv2d(out_dim*5, out_dim, 1, 1),
                                 nn.BatchNorm2d(out_dim),
                                 nn.ReLU(),
                                 nn.Conv2d(out_dim, out_dim, 1, 1),
                                 nn.MaxPool2d(2))
        
    def forward(self, x):
        
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.aspp5(x)
        _, _, w, h = x.shape
        x5 = F.upsample(x5, (w, h), mode='bilinear')

        cat = torch.cat([x1, x2, x3, x4, x5], 1)
        out = self.out(cat)

        return out 
    
if __name__ == "__main__":

    aspp = ASPP(in_dim=3, out_dim=32, d_size=12)

    input = torch.rand([2, 3, 256, 1248])
    aspp(input)
    print()
    torch.onnx.export(aspp, input, 'aspp.onnx')
