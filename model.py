
# import sys, os
# sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from models.encoders import KSC2022 as KSC2022_2d
from models.encoders import Segformer_Encorder
from models.encoders import SalsaNext as salsa_encoder
from models.decoders import * 
from models.decoders import SalsaNeXt as salsa_decoder 
from models.encoders_fusion import KSC2022 as KSC2022_fusion

import torch
import torch.nn as nn
import torch.nn.functional as F


class SalsaNeXt(nn.Module):
    def __init__(self, n_class):
        super(SalsaNeXt, self).__init__()
        self.n_class = n_class
        self.encoder = salsa_encoder()
        self.decoder = salsa_decoder(self.n_class) 
        
        
    def forward(self, x):
        
        down0b, down1b, down2b, down3b, down5c  = self.encoder(x)
        
        logits = self.decoder(down0b, down1b, down2b, down3b, down5c )
        
        logits = F.softmax(logits, dim=1)
        
        return logits
    
    
    

class KSC2022_Fusion(nn.Module):
    def __init__(self, input_shape, fusion_lev):
        super(KSC2022_Fusion, self).__init__()
        self.input_shape = input_shape
        self.fusion_lev = fusion_lev
        self.num_cls = 20

        self.encoder = KSC2022_fusion(self.input_shape, self.fusion_lev)
        self.decoder = TransUNet_Decorder(self.fusion_lev)

        self.Convolution = nn.Conv2d(in_channels=64, out_channels=self.num_cls, kernel_size=1)

    def forward(self, rgb, rem):
        if self.fusion_lev == "late":

            f1_rgb, f1_rem, f2_rgb, f2_rem, f3_rgb, f3_rem, f4_rgb, f4_rem = self.encoder(rgb, rem)
            x_RGB, x_REM = self.decoder([f1_rgb, f1_rem],[f2_rgb, f2_rem],[f3_rgb, f3_rem],[f4_rgb, f4_rem])
            segment_out_RGB = F.softmax(self.Convolution(x_RGB), dim=1)
            segment_out_REM = F.softmax(self.Convolution(x_REM), dim=1)
            segment_out = torch.mean(torch.stack([segment_out_RGB, segment_out_REM]),0)
        else:
            f1, f2, f3, f4 = self.encoder(rgb, rem)
            #
            # f1    128     48  160      
            # f2    512     24  80 
            # f3    1024    12  40
            # f4    2048    6   20

            x = self.decoder(f1, f2, f3, f4)
            segment_out = F.softmax(self.Convolution(x), dim=1)

        return segment_out

class KSC2022(nn.Module):
    def __init__(self, input_shape, fusion_lev, n_class):
        super(KSC2022, self).__init__()
        self.input_shape = input_shape
        self.fusion_lev = fusion_lev
        self.num_cls = n_class

        self.encoder = KSC2022_2d(self.input_shape, self.fusion_lev)
        self.decoder = TransUNet_Decorder(self.fusion_lev)
        # self.decoder = TransUNet_101(self.fusion_lev)
        # self.decoder = MLA_add(self.num_cls)
        # self.decoder = MLA_cat(self.num_cls)
        # self.decoder = PUP()
        # self.decoder = UNet_MLA_cat(self.num_cls)

        self.Convolution = nn.Conv2d(in_channels=64, out_channels=self.num_cls, kernel_size=1)

    def forward(self, input):

        f2, f3, f4, f5 = self.encoder(input, first_layer="ResNeSt")
        # f1, f2, f3, f4 = self.encoder(input, first_layer="segnext") 


        x = self.decoder(f2, f3, f4, f5)
        # x = self.decoder(f4)
        segment_out = F.softmax(self.Convolution(x), dim=1)

        return segment_out

class KSC2022_segformer(nn.Module): # for fusion 
    def __init__(self, input_shape, fusion_lev, n_class):
        """mid fusion in stage 2"""
        super(KSC2022_segformer, self).__init__()
        self.input_shape = input_shape
        self.fusion_lev = fusion_lev
        self.num_cls = n_class
        self.embed_dim = [64, 128, 320, 512]

        self.encoder = Segformer_Encorder(self.input_shape, self.fusion_lev)
        self.decoder = MLP_SegFormer(self.num_cls, 768, self.embed_dim, self.fusion_lev)
        self.Convolution = nn.Conv2d(in_channels=768, out_channels=self.num_cls, kernel_size=1)

    def forward(self, input):

        f1_img, f1_rem, f2_img, f2_rem, f3, f4 = self.encoder(input) 
        x = self.decoder(f1_img, f1_rem, f2_img, f2_rem, f3, f4)
        segment_out = F.softmax(self.Convolution(x), dim=1)
        return segment_out

if __name__ == "__main__":
    model = KSC2022(input_shape=(384//4, 1280//4), fusion_lev="none", n_class=20)
    from torchsummaryX import summary
    summary(model, torch.rand((5, 3, 384//4, 1280//4)))    
    from torchinfo import summary
    summary(model, (5, 3, 384//4, 1280//4))    
    

    # visualized model 
    from torchviz import make_dot
    from torch.autograd import Variable
    x = Variable(torch.randn((5, 3, 384//4, 1280//4))).cuda()
    make_dot(model(x), params=dict(model.named_parameters())).render("graph", format="png")


    # model = KSC2022_Fusion(input_shape=(384//4, 1280//4), fusion_lev="mid_stage2")
    # summary(model, torch.rand((5, 3, 384//4, 1280//4)), torch.rand((5, 1, 384//4, 1280//4)))    
    
    # model = KSC2022_Encorder(input_size=(1024//4,2048//4), fusion_lev="none") # transunet_encoder x.shape[1]
    # summary(model, torch.rand((5, 3, 1024//4, 2048//4)))



