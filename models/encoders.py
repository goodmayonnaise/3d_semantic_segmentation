import torch.nn as nn

from models.salsanext import ResContextBlock, ResBlock
from models.resnest import ResNeSt,  Bottleneck
from models.segnext import SpatialAttention
from models.segformer import *
from models.doubleunet import *


class DoubleUNet(nn.Module):
    def __init__(self, phase="1st"):
        super(DoubleUNet, self).__init__()
        if phase == "1st":
            self.enc1 = VGGBlock(3, 64, 64, True)
            self.enc2 = VGGBlock(64, 128, 128, True)
            self.enc3 = VGGBlock(128, 256, 256, True)
            self.enc4 = VGGBlock(256, 512, 512, True)
            self.enc5 = VGGBlock(512, 512, 512, True)
            
            # apply pretrained vgg19 weights on 1st unet
            vgg19 = torch.hub.load('pytorch/vision:v0.10.0', 'vgg19_bn', pretrained=True)
            
            self.enc1.conv1.weights = vgg19.features[0].weight
            self.enc1.bn1.weights = vgg19.features[1].weight
            self.enc1.conv2.weights = vgg19.features[3].weight
            self.enc1.bn2.weights = vgg19.features[4].weight
            self.enc2.conv1.weights = vgg19.features[7].weight
            self.enc2.bn1.weights = vgg19.features[8].weight
            self.enc2.conv2.weights = vgg19.features[10].weight
            self.enc2.bn2.weights = vgg19.features[11].weight
            self.enc3.conv1.weights = vgg19.features[14].weight
            self.enc3.bn1.weights = vgg19.features[15].weight
            self.enc3.conv2.weights = vgg19.features[17].weight
            self.enc3.bn2.weights = vgg19.features[18].weight
            self.enc4.conv1.weights = vgg19.features[27].weight
            self.enc4.bn1.weights = vgg19.features[28].weight
            self.enc4.conv2.weights = vgg19.features[30].weight
            self.enc4.bn2.weights = vgg19.features[31].weight
            self.enc5.conv1.weights = vgg19.features[33].weight
            self.enc5.bn1.weights = vgg19.features[34].weight
            self.enc5.conv2.weights = vgg19.features[36].weight
            self.enc5.bn2.weights = vgg19.features[37].weight
            del vgg19
            
        elif phase == "2nd":
            self.enc1 = VGGBlock(3, 64, 64, True, True)
            self.enc2 = VGGBlock(64, 128, 128, True, True)
            self.enc3 = VGGBlock(128, 256, 256, True, True)
            self.enc4 = VGGBlock(256, 512, 512, True, True)
            self.enc5 = VGGBlock(512, 512, 512, True, True)

        self.aspp = ASPP(512, 512)
        
    def forward(self, x):
        y_enc1 = self.enc1(x)
        y_enc2 = self.enc2(y_enc1)
        y_enc3 = self.enc3(y_enc2)
        y_enc4 = self.enc4(y_enc3)
        y_enc5 = self.enc5(y_enc4)

        # aspp bridge
        y_aspp = self.aspp(y_enc5)
        
        return y_aspp, y_enc4, y_enc3, y_enc2, y_enc1



class SalsaNext(nn.Module):
    def __init__(self):
        super(SalsaNext, self).__init__()

        self.downCntx = ResContextBlock(3, 32)
        self.downCntx2 = ResContextBlock(32, 32)
        self.downCntx3 = ResContextBlock(32, 32)

        self.resBlock1 = ResBlock(32, 2 * 32, 0.2, pooling=True, drop_out=False)
        self.resBlock2 = ResBlock(2 * 32, 2 * 2 * 32, 0.2, pooling=True)
        self.resBlock3 = ResBlock(2 * 2 * 32, 2 * 4 * 32, 0.2, pooling=True)
        self.resBlock4 = ResBlock(2 * 4 * 32, 2 * 4 * 32, 0.2, pooling=True)
        self.resBlock5 = ResBlock(2 * 4 * 32, 2 * 4 * 32, 0.2, pooling=False)

    def forward(self, x):
        downCntx = self.downCntx(x)             # 40    32  96  320
        downCntx = self.downCntx2(downCntx)     # 40    32  96  320
        downCntx = self.downCntx3(downCntx)     # 40    32  96  320

        down0c, down0b = self.resBlock1(downCntx)
        down1c, down1b = self.resBlock2(down0c)
        down2c, down2b = self.resBlock3(down1c)
        down3c, down3b = self.resBlock4(down2c)
        down5c = self.resBlock5(down3c)

        return down0b, down1b, down2b, down3b, down5c 

class KSC2022(nn.Module): 
    def __init__(self, input_shape, fusion_lev):
        super(KSC2022, self).__init__()

        self.input_shape = input_shape
        self.fusion_lev = fusion_lev

        # ResNeSt layer  -------------------------------------------------------------------------------------------------------------------------------
        # resnest 50 -----------------------------------------------------------
        self.resnest = ResNeSt(Bottleneck, [3, 4, 6, 3], # ------------------ 2x1s64d
                                radix=2, groups=1, bottleneck_width=64,
                                deep_stem=True, stem_width=32, avg_down=True,
                                avd=True, avd_first=False)
        # self.resnest = ResNeSt(Bottleneck, [3, 4, 6, 3], # ------------------ 2x2s40d
        #                         radix=2, groups=2, bottleneck_width=40,
        #                         deep_stem=True, stem_width=32, avg_down=True,
        #                         avd=True, avd_first=False)
        self.spartialattention_x1 = SpatialAttention(d_model=64)
        self.spartialattention_x2 = SpatialAttention(d_model=256)
        self.spartialattention_x3 = SpatialAttention(d_model=512)
        self.spartialattention_x4 = SpatialAttention(d_model=1024)
        self.spartialattention_x5 = SpatialAttention(d_model=2048)

        # # resnest 101 --------------------------------------------------------
        # self.resnest = ResNeSt(Bottleneck, [3, 4, 23, 3], # ------------------ 2x1s64d
        #                         radix=2, groups=1, bottleneck_width=64,
        #                         deep_stem=True, stem_width=64, avg_down=True,
        #                         avd=True, avd_first=False)
        # self.resnest = ResNeSt(Bottleneck, [3, 4, 23, 3], # ------------------ 2x2s40d
        #                         radix=2, groups=2, bottleneck_width=40,
        #                         deep_stem=True, stem_width=64, avg_down=True,
        #                         avd=True, avd_first=False)
        # # resnest 200 --------------------------------------------------------
        # self.resnest = ResNeSt(Bottleneck, [3, 24, 36, 3], # ------------------ 2x1s64d
        #                         radix=2, groups=1, bottleneck_width=64,
        #                         deep_stem=True, stem_width=64, avg_down=True,
        #                         avd=True, avd_first=False)
        # self.resnest = ResNeSt(Bottleneck, [3, 24, 36, 3], # ------------------ 2x2s40d
        #                         radix=2, groups=2, bottleneck_width=40,
        #                         deep_stem=True, stem_width=64, avg_down=True,
        #                         avd=True, avd_first=False)


        # # resnest 269 --------------------------------------------------------
        # self.resnest = ResNeSt(Bottleneck, [3, 30, 48, 8], # ------------------ 2x1s64d
        #                         radix=2, groups=1, bottleneck_width=64,
        #                         deep_stem=True, stem_width=64, avg_down=True,
        #                         avd=True, avd_first=False)
        # self.resnest = ResNeSt(Bottleneck, [3, 30, 48, 8], # ------------------ 2x2s40d
        #                         radix=2, groups=2, bottleneck_width=40,
        #                         deep_stem=True, stem_width=64, avg_down=True,
        #                         avd=True, avd_first=False)
        
        # self.spartialattention_x1 = SpatialAttention(d_model=128)
        # self.spartialattention_x2 = SpatialAttention(d_model=256)
        # self.spartialattention_x3 = SpatialAttention(d_model=512)
        # self.spartialattention_x4 = SpatialAttention(d_model=1024)
        
        self.spartialattention = SpatialAttention(d_model=3)

        # for SegFormer ------------------------------------------------------------------------------
        self.embed_dim = 512
        self.patch_sizes = 3 
        self.padding_sizes = 1 
        self.strides = 2
        self.reduction_ratio = 1
        self.num_heads = 8
        self.expantion_ratio = 4
        self.num_tf = 9
        self.drop_path = 0.1

        self.patch_layer1 = PatchEmbedding_SegFormer(64, self.patch_sizes, self.padding_sizes, self.strides, 3)
        self.tfblock1 = EfficientAttentionBlock((self.input_shape[0]//2, self.input_shape[1]//2), 64, self.strides, self.reduction_ratio, self.num_heads, self.expantion_ratio, drop_path=self.drop_path)
        self.transform1 = Rearrange('b (h w) e -> b e h w', h=self.input_shape[0]//2, w=self.input_shape[1]//2)
        
        self.patch_layer2 = PatchEmbedding_SegFormer(256, self.patch_sizes, self.padding_sizes, self.strides, 64)
        self.tfblock2 = EfficientAttentionBlock((self.input_shape[0]//4, self.input_shape[1]//4), 256, self.strides, self.reduction_ratio, self.num_heads, self.expantion_ratio, drop_path=self.drop_path)
        self.transform2 = Rearrange('b (h w) e -> b e h w', h=self.input_shape[0]//4, w=self.input_shape[1]//4)

        self.patch_layer3 = PatchEmbedding_SegFormer(512, self.patch_sizes, self.padding_sizes, self.strides, 256)
        self.tfblock3 = EfficientAttentionBlock((self.input_shape[0]//8, self.input_shape[1]//8), 512, self.strides, self.reduction_ratio, self.num_heads, self.expantion_ratio, drop_path=self.drop_path)
        self.transform3 = Rearrange('b (h w) e -> b e h w', h=self.input_shape[0]//8, w=self.input_shape[1]//8)

        self.patch_layer4 = PatchEmbedding_SegFormer(1024, self.patch_sizes, self.padding_sizes, self.strides, 512)
        self.tfblock4 = EfficientAttentionBlock((self.input_shape[0]//16, self.input_shape[1]//16), 1024, self.strides, self.reduction_ratio, self.num_heads, self.expantion_ratio, drop_path=self.drop_path)
        self.transform4 = Rearrange('b (h w) e -> b e h w', h=self.input_shape[0]//16, w=self.input_shape[1]//16)


    def forward(self, input, first_layer="ResNeSt"):
        if first_layer == "ResNeSt":
            
            f1 = self.resnest.conv1(input)
            f1 = self.resnest.bn1(f1)
            f1 = self.resnest.relu(f1)
            
            f2 = self.resnest.maxpool(f1)
            f2 = self.resnest.layer1(f2)
            
            f3 = self.resnest.layer2(f2)
            
            f4 = self.resnest.layer3(f3)
            
            f5 = self.resnest.layer4(f4)
            
            # f1 = self.spartialattention_x1(f1)
            f2 = self.spartialattention_x2(f2)
            f3 = self.spartialattention_x3(f3)
            f4 = self.spartialattention_x4(f4)
            f5 = self.spartialattention_x5(f5)

        else: # segnext first
            x = self.spartialattention(input)
            f1, f2, f3, f4 = self.resnest(x)
        
        return f2, f3, f4, f5

