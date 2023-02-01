from einops.layers.torch import Rearrange
import torch
import torch.nn as nn
from torch import Tensor

class PatchEmbedding_SegFormer(nn.Module):
    def __init__(self, embed_dims, patch_size, padding_size, stride, in_ch):
        super(PatchEmbedding_SegFormer, self).__init__()
        self.embed_dims = embed_dims
        self.patch_size = patch_size
        self.padding_size = padding_size
        self.strid = stride
        self.in_ch = in_ch

        self.conv = nn.Conv2d(in_channels=self.in_ch, out_channels=self.embed_dims, kernel_size=self.patch_size, stride=self.strid, padding=(self.padding_size, self.padding_size))
        self.transform = Rearrange('b e w h  -> b (h w) e')
        self.layerNorm = nn.LayerNorm(self.embed_dims)

    def forward(self, x):
        x = self.conv(x)
        x = self.transform(x)
        x = self.layerNorm(x)
        return x

class MixFFN_SegFormer(nn.Module):
    def __init__(self, embed_dim, expansion_factor):
        super(MixFFN_SegFormer, self).__init__()
        self.embed_dim = embed_dim
        self.expansion_factor = expansion_factor
        self.hidden_dim = self.embed_dim * self.expansion_factor

        # layer
        self.conv = nn.Conv2d(in_channels=self.embed_dim, out_channels=self.hidden_dim, kernel_size=1)
        self.DepthwiseConv = nn.Conv2d(groups=self.hidden_dim, in_channels=self.hidden_dim, out_channels=self.hidden_dim, kernel_size=3, padding=1)
        self.actlayer = nn.GELU()
        self.drop1 = nn.Dropout(0.3)
        self.drop2 = nn.Dropout(0.3)
        self.conv2 = nn.Conv2d(in_channels=self.hidden_dim, out_channels=self.embed_dim, kernel_size=1)
        self.addlayer = torch.add

    def forward(self, x):
        x = self.conv(x)
        x = self.DepthwiseConv(x)
        x = self.actlayer(x)
        x = self.drop1(x)
        x = self.conv2(x)
        x = self.drop2(x)
        return x

class SpatialReduction_SegFormer(nn.Module):
    def __init__(self, embed_dims, reduction_ratio, feat_size, num_heads):
        super(SpatialReduction_SegFormer, self).__init__()
        self.embed_dims = embed_dims
        self.reduction_ratio = reduction_ratio
        self.feat_size = feat_size # 
        self.num_heads = num_heads
        # layer
        self.transform = Rearrange('b (h w) e -> b e h w', h=feat_size[0], w=feat_size[1]) # ??
        self._SR = nn.Conv2d(in_channels=self.embed_dims, out_channels=self.embed_dims, kernel_size=self.reduction_ratio, stride=self.reduction_ratio)#수정 완료
        self.transform2 = Rearrange('b e h w -> b (h w) e')
        self.layerNorm = nn.LayerNorm(self.embed_dims)
        self._kv = nn.Linear(in_features=self.embed_dims, out_features=self.embed_dims*2, bias=False)#수정 완료

    def forward(self, x):              # x 2 128 512
        b, n, c = x.shape 
        x = self.transform(x)          # b c h w
        x = self._SR(x)                # b c h/sr w/sr
        x = self.transform2(x)         # b n/(sr*sr) c
        x = self.layerNorm(x)
        kv = self._kv(x).reshape(b,-1, 2,self.num_heads, c//self.num_heads).permute(2,0,3,1,4)# 2 b num_head n/(sr*sr) c//num_head
        k, v = kv.chunk(2, dim=0) # b num_head n/(sr*sr) c//num_head
        return torch.squeeze(k,0), torch.squeeze(v,0) 

class EfficientAttentionBlock(nn.Module):
    def __init__(self, input_size, embed_dim, stride, reduction_ratio, num_heads, expantion_ratio, drop_path):
        super(EfficientAttentionBlock, self).__init__()
        self.input_size = input_size
        self.embed_dim = embed_dim
        self.stride = stride
        self.reduction_ratio = reduction_ratio
        self.num_heads = num_heads
        self.expantion_ratio = expantion_ratio
        self.drop_path = drop_path

        self.layerNorm = nn.LayerNorm(self.embed_dim)
        self._q = nn.Linear(in_features=self.embed_dim, out_features=self.embed_dim, bias=False)
        self.SRlayer = SpatialReduction_SegFormer(embed_dims=self.embed_dim, reduction_ratio=self.reduction_ratio, feat_size=self.input_size, num_heads = self.num_heads)
        self.SAlayer = MultiHeadSelfAttentionBlock(self.embed_dim, self.num_heads)
        self.drop_path = DropPath(self.drop_path) if self.drop_path > 0. else nn.Identity()
        self.transform = Rearrange('b (h w) c -> b c h w', h=self.input_size[0], w=self.input_size[1])
        self.MixFFN = MixFFN_SegFormer(embed_dim=self.embed_dim, expansion_factor=self.expantion_ratio)        
        self.tf1 = Rearrange('b c h w  -> b (h w) c ', h=self.input_size[0], w=self.input_size[1])      
        self.layerNorm1 = nn.LayerNorm(self.embed_dim, self.embed_dim)
        
    def forward(self, x):
        # print('x size',x.size())
        # print('x shape',x.shape)
        i = self.layerNorm(x)   # 3 128 512
        ############################## 여기부터가 segformer논문 그림 2의 Efficient self attention ###############################
        q = self._q(i)  # 3 128 512 # q 는 b n c 
        k, v = self.SRlayer(i) # k,v는 b num_head n/(sr*sr) c//num_head
        atten = self.SAlayer(q, k, v)
        i = i + self.drop_path(atten)
        ############################## 여기까지가 segformer논문 그림 2의 Efficient self attention ###############################
        m = self.layerNorm1(i)
        m = self.transform(m)        
        m = self.MixFFN(m)
        i = i + self.tf1(m)
        return i

class Segformer_Encorder(nn.Module): 
    def __init__(self, input_size, layers_channel:list=[3, 64, 128, 320, 512], extract_feature_map:int = 2):    
        super(Segformer_Encorder, self).__init__()

        self.input_size = input_size
        self.channels_inputs = layers_channel[0]
        self.extract_feature_map = extract_feature_map

        # for SegFormer 
        self.input_shapes = [(self.input_size[0]//(v//self.extract_feature_map), self.input_size[1]//(v//self.extract_feature_map)) for v in [4, 8, 16, 32]]
        # self.input_shapes = [(self.input_size[0]//v, self.input_size[1]//v) for v in ([4, 8, 16, 32] if self.extract_feature_map==1 else [2, 4, 8, 16])]
        self.embed_dim = layers_channel[1:]   # [64, 128, 320, 512]
        self.patch_sizes     = [7, 3, 3, 3] if self.extract_feature_map==1 else [3, 3, 3, 3]         # stage 2이후 부터는 전부 3
        self.padding_sizes   = [3, 1, 1, 1] if self.extract_feature_map==1 else [1, 1, 1, 1]        # stage 2이후 부터는 전부 1
        self.strides         = [4, 2, 2, 2] if self.extract_feature_map==1 else [2, 2, 2, 2]          # stage 2이후 부터는 전부 2
        self.reduction_ratio = [8, 4, 2, 1]   # attention layer에서 사용될 reduction 비율
        self.num_heads       = [1, 2, 5, 8] 
        self.expantion_ratio = [4, 4, 4, 4]   # dimension 확장 비율(in Mix_FFN)
        self.num_tf          = [3, 6, 40, 3]
        self.drop_path       = 0.1
        
        self.patch_embedding1 = PatchEmbedding_SegFormer(self.embed_dim[0], self.patch_sizes[0], self.padding_sizes[0], self.strides[0], self.channels_inputs)
        self.patch_embedding2 = PatchEmbedding_SegFormer(self.embed_dim[1], self.patch_sizes[1], self.padding_sizes[1], self.strides[1], self.embed_dim[0])
        self.patch_embedding3 = PatchEmbedding_SegFormer(self.embed_dim[2], self.patch_sizes[2], self.padding_sizes[2], self.strides[2], self.embed_dim[1])
        self.patch_embedding4 = PatchEmbedding_SegFormer(self.embed_dim[3], self.patch_sizes[3], self.padding_sizes[3], self.strides[3], self.embed_dim[2])

        self.tf_block1 = EfficientAttentionBlock(self.input_shapes[0], self.embed_dim[0], self.strides, self.reduction_ratio[0], self.num_heads[0], self.expantion_ratio[0], drop_path=self.drop_path)
        self.tf_block2 = EfficientAttentionBlock(self.input_shapes[1], self.embed_dim[1], self.strides, self.reduction_ratio[1], self.num_heads[1], self.expantion_ratio[1], drop_path=self.drop_path)        
        self.tf_block3 = EfficientAttentionBlock(self.input_shapes[2], self.embed_dim[2], self.strides, self.reduction_ratio[2], self.num_heads[2], self.expantion_ratio[2], drop_path=self.drop_path)
        self.tf_block4 = EfficientAttentionBlock(self.input_shapes[3], self.embed_dim[3], self.strides, self.reduction_ratio[3], self.num_heads[3], self.expantion_ratio[3], drop_path=self.drop_path)

        self.transform1 = Rearrange('b (h w) e -> b e h w ', h=self.input_shapes[0][0], w=self.input_shapes[0][1], e=self.embed_dim[0])
        self.transform2 = Rearrange('b (h w) e -> b e h w ', h=self.input_shapes[1][0], w=self.input_shapes[1][1], e=self.embed_dim[1])
        self.transform3 = Rearrange('b (h w) e -> b e h w ', h=self.input_shapes[2][0], w=self.input_shapes[2][1], e=self.embed_dim[2])
        self.transform4 = Rearrange('b (h w) e -> b e h w ', h=self.input_shapes[3][0], w=self.input_shapes[3][1], e=self.embed_dim[3])   

    def forward(self, input):
        img = input
        # stage 1
        f1 = self.patch_embedding1(img)
        for _ in range(self.num_tf[0]):
            f1 = self.tf_block1(f1)
        #f1 = Layernorm(f1) 추가되야함
        f1 = self.transform1(f1)

        # stage 2
        f2 = self.patch_embedding2(f1)
        for _ in range(self.num_tf[1]):
            f2 = self.tf_block2(f2)
        
        f2 = self.transform2(f2)

        # stage 3
        f3 = self.patch_embedding3(f2)
        for _ in range(self.num_tf[2]):
            f3 = self.tf_block3(f3)
        
        f3 = self.transform3(f3)
        
        f4 = self.patch_embedding4(f3)
        for _ in range(self.num_tf[3]-1):
            f4 = self.tf_block4(f4)
        
        f4 = self.transform4(f4)

        return f1, f2, f3, f4

class MLPDecoder_SegFormer(nn.Module):
    def __init__(self, n_class, hidden_dim, layer_channels, include_top, extract_feature_map):
        super(MLPDecoder_SegFormer, self).__init__()
        self.num_cls = n_class
        self.extract_feature_map =  extract_feature_map
        self.hidden_dim = hidden_dim
        self.embed_dim = layer_channels[1:]
        self.include_top = include_top
        # conv layer
        self.conv1 = nn.Conv2d(in_channels=self.embed_dim[0], out_channels=self.hidden_dim, kernel_size=1)
        self.conv2 = nn.Conv2d(in_channels=self.embed_dim[1], out_channels=self.hidden_dim, kernel_size=1)
        self.conv3 = nn.Conv2d(in_channels=self.embed_dim[2], out_channels=self.hidden_dim, kernel_size=1) 
        self.conv4 = nn.Conv2d(in_channels=self.embed_dim[3], out_channels=self.hidden_dim, kernel_size=1) 
        if self.include_top:
            self.Convolution = nn.Conv2d(in_channels=len(self.embed_dim)*self.hidden_dim, out_channels=self.num_cls, kernel_size=1) # segment 출력층
        # Upsample layer
        self.Upsample1 = nn.Upsample(scale_factor= self.extract_feature_map * 1 if self.extract_feature_map > 1 else 1, mode='bilinear')
        # self.Upsample1 = nn.Upsample(scale_factor= self.extract_feature_map * 1, mode='bilinear')
        self.Upsample2 = nn.Upsample(scale_factor= self.extract_feature_map * 2, mode='bilinear')
        self.Upsample3 = nn.Upsample(scale_factor= self.extract_feature_map * 4, mode='bilinear')
        self.Upsample4 = nn.Upsample(scale_factor= self.extract_feature_map * 8, mode='bilinear')

        # concat layer
        self.concat = torch.cat

    def forward(self, f1, f2, f3, f4):
        x_list = list()
        x_list.append(self.Upsample1(self.conv1(f1)) if self.extract_feature_map > 1 else self.conv1(f1))
        x_list.append(self.Upsample2(self.conv2(f2)))
        x_list.append(self.Upsample3(self.conv3(f3)))
        x_list.append(self.Upsample4(self.conv4(f4)))
        x = self.concat(x_list,dim=1)
        
        if self.include_top:            
            x = self.Convolution(x)

        return x

class MultiHeadSelfAttentionBlock(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(MultiHeadSelfAttentionBlock, self).__init__()
        self.num_heads = num_heads
        self.embed_dim = embed_dim
 
        self.atten_drop = nn.Dropout(0.3)
        self.proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.proj_drop = nn.Dropout(0.3)

    def forward(self, q, k, v):# q 는 b n c | k,v는 b num_head n/(sr*sr) c//num_head
        b, n, c = q.shape         
        q = q.reshape(b, n, self.num_heads, c//self.num_heads).permute(0,2,1,3) # b num_nead n c//num_head        
        atten = q @ k.transpose(-2,-1)
        atten = atten.softmax(dim=-1)
        atten = self.atten_drop(atten)
        x = (atten @ v).transpose(1,2).reshape(b, n, c)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    Copied from timm
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    def __init__(self, p: float = None):
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        if self.p == 0. or not self.training:
            return x
        kp = 1 - self.p
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = kp + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # binarize
        return x.div(kp) * random_tensor

    



