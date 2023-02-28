

import torch.nn as nn 

class AttentionModule(nn.Module):
    def __init__(self, dim, k_size):
        super().__init__()

        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        
        self.attn_layers = []
        for k in k_size:
            self.attn_layers.append(nn.Sequential(
                nn.Conv2d(dim, dim, (1, k), padding=(0, int((k-1)/2)), groups=dim),
                nn.Conv2d(dim, dim, (k, 1), padding=(int((k-1)/2), 0), groups=dim)
                ))
        self.attn_layer = nn.Sequential(*self.attn_layers)

        self.conv = nn.Conv2d(dim, dim, 1)

    def forward(self, x, k_size):
        u = x.clone()
        attn = self.conv0(x) 

        attn_list = []
        for k in range(k_size):
            attn_list.append(self.attn_layers[k](attn))
            
        attn = attn + sum(attn_list)
        attn = self.conv(attn)

        return attn * u

class DilatedAttentionModule(nn.Module):
    def __init__(self, dim, k_size):
        super().__init__()

        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)

        self.attn_layers = []
        for k in k_size:
            self.attn_layers.append(nn.Sequential(
                nn.Conv2d(dim, dim, (1, k), padding=(0, int((k-1)/2)), groups=dim, dilation=(int((k-1)/2), 1)),
                nn.Conv2d(dim, dim, (k, 1), padding=(int((k-1)/2), 0), groups=dim, dilation=(1, int((k-1)/2)))
            ))
        self.attn_layers = nn.Sequential(*self.attn_layers)

        self.conv = nn.Conv2d(dim, dim, 1)

    def forward(self, x, k_size):
        u = x.clone()
        attn = self.conv0(x) # 256 64 128

        attn_list = []
        for k in range(k_size):
            attn_list.append(self.attn_layers[k](attn))

        attn = attn + sum(attn_list)
        attn = self.conv(attn)

        return attn * u

class SpatialAttention(nn.Module):
    def __init__(self, d_model, k_scale):
        super().__init__()
        self.d_model = d_model
        self.proj_1 = nn.Conv2d(d_model, d_model, 1)
        self.activation = nn.GELU()
        # self.spatial_gating_unit = AttentionModule(d_model, k_scale)
        self.spatial_gating_unit = DilatedAttentionModule(d_model, k_scale)
        self.proj_2 = nn.Conv2d(d_model, d_model, 1)

    def forward(self, x, k_size):
        shorcut = x.clone()
        x = self.proj_1(x)
        x = self.activation(x)
        x = self.spatial_gating_unit(x, k_size)
        x = self.proj_2(x)
        x = x + shorcut
        return x


        

