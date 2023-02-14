import torch.nn as nn
from models.resnest import ResNeSt,  Bottleneck
from models.segnext import SpatialAttention
from modules import Concatenate
from models.segformer import *


class KSC2022(nn.Module): 
    def __init__(self, input_shape, fusion_lev):
        super(KSC2022, self).__init__()

        self.input_shape = input_shape
        self.fusion_lev = fusion_lev

        # ResNeSt layer  -------------------------------------------------------------------------------------------------------------------------------
        # resnest 50 -----------------------------------------------------------
        self.resnest = ResNeSt(Bottleneck, [3, 4, 6, 3], # ------------------ 2x2s40d
                                radix=2, groups=1, bottleneck_width=64,
                                deep_stem=True, stem_width=32, avg_down=True,
                                avd=True, avd_first=False)
        self.spartialattention_x1 = SpatialAttention(d_model=64)
        self.spartialattention_x2 = SpatialAttention(d_model=256)
        self.spartialattention_x3 = SpatialAttention(d_model=512)
        self.spartialattention_x4 = SpatialAttention(d_model=1024)
        # self.spartialattention_x5 = SpatialAttention(d_model=2048)

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
            
            f4 = self.patch_layer4(f3)
            for _ in range(self.num_tf):
                f4 = self.tfblock4(f4)
            f4 = self.transform4(f4)
            
            f1 = self.spartialattention_x1(f1)
            f2 = self.spartialattention_x2(f2)
            f3 = self.spartialattention_x3(f3)
            f4 = self.spartialattention_x4(f4)

        else: # segnext first
            x = self.spartialattention(input)
            f1, f2, f3, f4 = self.resnest(x)
        
        return f1, f2, f3, f4
