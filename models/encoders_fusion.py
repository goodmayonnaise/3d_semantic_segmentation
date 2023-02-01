
from models.resnest import Fusion_ResNeSt, Bottleneck
from models.segformer import PatchEmbedding_SegFormer, EfficientAttentionBlock, Rearrange
from modules import Concatenate

import torch.nn as nn
import torch 


class KSC2022(nn.Module): 
    def __init__(self, input_size, fusion_lev):
        super(KSC2022_Fusion_Encorder, self).__init__()

        self.input_size = input_size
        self.fusion_lev = fusion_lev

        # for SegFormer 
        self.embed_dim = 512
        self.patch_sizes = 3 
        self.padding_sizes = 1 
        self.strides = 2
        self.reduction_ratio = 2
        self.num_heads = 5
        self.expantion_ratio = 4
        self.num_tf = 9
        # define layer 
        self.concat = Concatenate(axis=1)

        self.resnest = Fusion_ResNeSt(Bottleneck, [3, 4, 6, 3],
                                      radix=2, groups=2, bottleneck_width=40,
                                      deep_stem=True, stem_width=32, avg_down=True,
                                      avd=True, avd_first=False, fusion_lev=self.fusion_lev)            
        # patch embedding layer 
        if self.fusion_lev == "mid_stage3":
            self.patch_layer = PatchEmbedding_SegFormer(self.embed_dim, self.patch_sizes, self.padding_sizes, self.strides, self.embed_dim*2)
        elif self.fusion_lev in ["mid_stage4", "late"]:  
            self.patch_layer_RGB = PatchEmbedding_SegFormer(self.embed_dim, self.patch_sizes, self.padding_sizes, self.strides, self.embed_dim)
            self.patch_layer_REM = PatchEmbedding_SegFormer(self.embed_dim, self.patch_sizes, self.padding_sizes, self.strides, self.embed_dim) 
        else:
            self.patch_layer = PatchEmbedding_SegFormer(self.embed_dim, self.patch_sizes, self.padding_sizes, self.strides, self.embed_dim)
        
        # transformer block layer
        if self.fusion_lev in ["mid_stage4", "late"]: 
            self.tfblock_RGB = EfficientAttentionBlock((self.input_size[0]//16, self.input_size[1]//16), self.embed_dim, self.strides, self.reduction_ratio, self.num_heads, self.expantion_ratio)
            self.tfblock_REM = EfficientAttentionBlock((self.input_size[0]//16, self.input_size[1]//16), self.embed_dim, self.strides, self.reduction_ratio, self.num_heads, self.expantion_ratio)
        else:
            self.tfblock = EfficientAttentionBlock((self.input_size[0]//16, self.input_size[1]//16), self.embed_dim, self.strides, self.reduction_ratio, self.num_heads, self.expantion_ratio)

        self.transform = Rearrange('b e h w   -> b (h w) e', e=self.embed_dim)
        self.transform1 = Rearrange('b e h w   -> b (h w) e', e=self.embed_dim)

    def forward(self, rgb, rem):
        f1, f2, f3, f4 = self.resnest(rgb, rem)
        
        # f1, f2, f3 = self.resnest(input)
        # f4 = self.patch_layer(f3) 
        # for i in range(self.num_tf):
        #     f4 = self.tfblock(f4)  
        #     if i != 0 or i != 8:
        #         f4 = self.transform(f4) 
        # f4 = self.tfblock(f4) # 512 24 78 # in : b n c  out : b h w c  #--------완전 이상함------------#--------------------#--------------------#--------------------#--------------------#--------------------#--------------------#--------------------#--------------------#--------------------#--------------------#--------------------#--------------------
        
        return f1, f2, f3, f4 


class Segformer(nn.Module): 
    def __init__(self, input_size, fusion_lev):
        super(Segformer_Encorder, self).__init__()

        self.input_size = input_size
        self.fusion_lev = fusion_lev

        # for SegFormer 
        self.embed_dim = [64, 128, 320, 512]
        self.patch_sizes = [7, 3] # stage 2이후 부터는 전부 3
        self.padding_sizes = [3, 1] # stage 2이후 부터는 전부 1
        self.strides = [4, 2] # stage 2이후 부터는 전부 2
        self.reduction_ratio = [8, 4, 2, 1] #attention layer에서 사용될 reduction 비율
        self.num_heads = [1, 2, 5, 8] 
        self.expantion_ratio = [4, 4, 4, 4] #dimention 확장 비율(in Mix_FFN)
        self.num_tf = [3, 6, 40, 3]

        self.patch_embedding1 = {"img": PatchEmbedding_SegFormer(self.embed_dim[0], self.patch_sizes[0], self.padding_sizes[0], self.strides[0], 3), "rem": PatchEmbedding_SegFormer(self.embed_dim[0], self.patch_sizes[0], self.padding_sizes[0], self.strides[0], 1)}
        self.patch_embedding2 = {"img": PatchEmbedding_SegFormer(self.embed_dim[1], self.patch_sizes[1], self.padding_sizes[1], self.strides[1], self.embed_dim[0]), "rem": PatchEmbedding_SegFormer(self.embed_dim[1], self.patch_sizes[1], self.padding_sizes[1], self.strides[1], self.embed_dim[0])}
        self.patch_embedding3 = PatchEmbedding_SegFormer(self.embed_dim[2], self.patch_sizes[1], self.padding_sizes[1], self.strides[1], self.embed_dim[1]*2)
        self.patch_embedding4 = PatchEmbedding_SegFormer(self.embed_dim[3], self.patch_sizes[1], self.padding_sizes[1], self.strides[1], self.embed_dim[2])
        
        self.tf_block1 = {"img": EfficientAttentionBlock((self.input_size[0]//4, self.input_size[1]//4), self.embed_dim[0], self.reduction_ratio[0], self.num_heads[0], self.expantion_ratio[0]), "rem": EfficientAttentionBlock((self.input_size[0]//4, self.input_size[1]//4), self.embed_dim[0], self.reduction_ratio[0], self.num_heads[0], self.expantion_ratio[0])}
        self.tf_block2 = {"img": EfficientAttentionBlock((self.input_size[0]//8, self.input_size[1]//8), self.embed_dim[1], self.reduction_ratio[1], self.num_heads[1], self.expantion_ratio[1]), "rem": EfficientAttentionBlock((self.input_size[0]//8, self.input_size[1]//8), self.embed_dim[1], self.reduction_ratio[1], self.num_heads[1], self.expantion_ratio[1])}
        self.tf_block3 = EfficientAttentionBlock((self.input_size[0]//16, self.input_size[1]//16), self.embed_dim[2], self.reduction_ratio[2], self.num_heads[2], self.expantion_ratio[2])
        self.tf_block4 = EfficientAttentionBlock((self.input_size[0]//32, self.input_size[1]//32), self.embed_dim[3], self.reduction_ratio[3], self.num_heads[3], self.expantion_ratio[3])

        self.transform1 = Rearrange('b (h w) e -> b e h w ', h=self.input_size[0]//4, w=self.input_size[1]//4, e=self.embed_dim[0])
        self.transform2 = Rearrange('b (h w) e -> b e h w ', h=self.input_size[0]//8, w=self.input_size[1]//8,e=self.embed_dim[1])
        self.transform3 = Rearrange('b (h w) e -> b e h w ', h=self.input_size[0]//16, w=self.input_size[1]//16,e=self.embed_dim[2])
        self.transform4 = Rearrange('b (h w) e -> b e h w ', h=self.input_size[0]//32, w=self.input_size[1]//32,e=self.embed_dim[3])

        self.cat = torch.cat
    def forward(self, input):
        img = input[0]
        rem = input[1]
        # stage 1
        f1_img = self.patch_embedding1["img"](img)
        for i in range(self.num_tf[0]):
            f1_img = self.tf_block1["img"](f1_img)
        f1_img = self.transform1(f1_img)

        f1_rem = self.patch_embedding1["rem"](rem)
        for i in range(self.num_tf[0]):
            f1_rem = self.tf_block1["rem"](f1_rem)
        f1_rem = self.transform1(f1_rem)

        # stage 2
        f2_img = self.patch_embedding2["img"](f1_img)
        for i in range(self.num_tf[1]):
            f2_img = self.tf_block2["img"](f2_img)
        f2_img = self.transform2(f2_img)

        f2_rem = self.patch_embedding2["rem"](f1_rem)
        for i in range(self.num_tf[1]):
            f2_rem = self.tf_block2["rem"](f2_rem)
        f2_rem = self.transform2(f2_rem)

        f2_cat = self.cat([f2_img, f2_rem], dim=1)

        # stage 3
        f3 = self.patch_embedding3(f2_cat)
        for i in range(self.num_tf[2]):
            f3 = self.tf_block3(f3)
        f3 = self.transform3(f3)

        
        f4 = self.patch_embedding4(f3)
        for i in range(self.num_tf[3]-1):
            f4 = self.tf_block4(f4)
        f4 = self.transform4(f4)
        
        return f1_img, f1_rem, f2_img, f2_rem, f3, f4

