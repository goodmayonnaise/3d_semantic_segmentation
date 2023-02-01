

import torch.nn as nn 

class AttentionModule(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        # 3 7 11 (base) ----------------------------------------------------------------------------------
        self.conv0_1 = nn.Conv2d(dim, dim, (1,3), padding=(0,1), groups=dim)
        self.conv0_2 = nn.Conv2d(dim, dim, (3,1), padding=(1,0), groups=dim)
        self.conv1_1 = nn.Conv2d(dim, dim, (1,7), padding=(0,3), groups=dim)
        self.conv1_2 = nn.Conv2d(dim, dim, (7,1), padding=(3,0), groups=dim)
        self.conv2_1 = nn.Conv2d(dim, dim, (1,11), padding=(0,5), groups=dim)
        self.conv2_2 = nn.Conv2d(dim, dim, (11,1), padding=(5,0), groups=dim)

        # 3 5 7 ------------------------------------------------------------------------------------------
        # self.conv0_1 = nn.Conv2d(dim, dim, (1,3), padding=(0,1), groups=dim)
        # self.conv0_2 = nn.Conv2d(dim, dim, (3,1), padding=(1,0), groups=dim)
        # self.conv1_1 = nn.Conv2d(dim, dim, (1,5), padding=(0,2), groups=dim)
        # self.conv1_2 = nn.Conv2d(dim, dim, (5,1), padding=(2,0), groups=dim)
        # self.conv2_1 = nn.Conv2d(dim, dim, (1,7), padding=(0,3), groups=dim)
        # self.conv2_2 = nn.Conv2d(dim, dim, (7,1), padding=(3,0), groups=dim)

        # 5 7 9 ------------------------------------------------------------------------------------------
        # self.conv0_1 = nn.Conv2d(dim, dim, (1,5), padding=(0,2), groups=dim)
        # self.conv0_2 = nn.Conv2d(dim, dim, (5,1), padding=(2,0), groups=dim)
        # self.conv1_1 = nn.Conv2d(dim, dim, (1,7), padding=(0,3), groups=dim)
        # self.conv1_2 = nn.Conv2d(dim, dim, (7,1), padding=(3,0), groups=dim)
        # self.conv2_1 = nn.Conv2d(dim, dim, (1,9), padding=(0,4), groups=dim)
        # self.conv2_2 = nn.Conv2d(dim, dim, (9,1), padding=(4,0), groups=dim)

        # 7 9 11 ------------------------------------------------------------------------------------------
        # self.conv0_1 = nn.Conv2d(dim, dim, (1,7), padding=(0,3), groups=dim)
        # self.conv0_2 = nn.Conv2d(dim, dim, (7,1), padding=(3,0), groups=dim)
        # self.conv1_1 = nn.Conv2d(dim, dim, (1,9), padding=(0,4), groups=dim)
        # self.conv1_2 = nn.Conv2d(dim, dim, (9,1), padding=(4,0), groups=dim)
        # self.conv2_1 = nn.Conv2d(dim, dim, (1,11), padding=(0,5), groups=dim)
        # self.conv2_2 = nn.Conv2d(dim, dim, (11,1), padding=(5,0), groups=dim)

        # 7 11 21 -----------------------------------------------------------------------------------------
        # self.conv0_1 = nn.Conv2d(dim, dim, (1,7), padding=(0,3), groups=dim)
        # self.conv0_2 = nn.Conv2d(dim, dim, (7,1), padding=(3,0), groups=dim)
        # self.conv1_1 = nn.Conv2d(dim, dim, (1,11), padding=(0,5), groups=dim)
        # self.conv1_2 = nn.Conv2d(dim, dim, (11,1), padding=(5,0), groups=dim)
        # self.conv2_1 = nn.Conv2d(dim, dim, (1,21), padding=(0,10), groups=dim)
        # self.conv2_2 = nn.Conv2d(dim, dim, (21,1), padding=(10,0), groups=dim)

        # 3 5 7 9 ------------------------------------------------------------------------------------------
        # self.conv0_1 = nn.Conv2d(dim, dim, (1,3), padding=(0,1), groups=dim)
        # self.conv0_2 = nn.Conv2d(dim, dim, (3,1), padding=(1,0), groups=dim)
        # self.conv1_1 = nn.Conv2d(dim, dim, (1,5), padding=(0,2), groups=dim)
        # self.conv1_2 = nn.Conv2d(dim, dim, (5,1), padding=(2,0), groups=dim)
        # self.conv2_1 = nn.Conv2d(dim, dim, (1,7), padding=(0,3), groups=dim)
        # self.conv2_2 = nn.Conv2d(dim, dim, (7,1), padding=(3,0), groups=dim)
        # self.conv3_1 = nn.Conv2d(dim, dim, (1,9), padding=(0,4), groups=dim)
        # self.conv3_2 = nn.Conv2d(dim, dim, (9,1), padding=(4,0), groups=dim)

        # # 5 7 9 11 ------------------------------------------------------------------------------------------
        # self.conv0_1 = nn.Conv2d(dim, dim, (1,5), padding=(0,2), groups=dim)
        # self.conv0_2 = nn.Conv2d(dim, dim, (5,1), padding=(2,0), groups=dim)
        # self.conv1_1 = nn.Conv2d(dim, dim, (1,7), padding=(0,3), groups=dim)
        # self.conv1_2 = nn.Conv2d(dim, dim, (7,1), padding=(3,0), groups=dim)
        # self.conv2_1 = nn.Conv2d(dim, dim, (1,9), padding=(0,4), groups=dim)
        # self.conv2_2 = nn.Conv2d(dim, dim, (9,1), padding=(4,0), groups=dim)
        # self.conv3_1 = nn.Conv2d(dim, dim, (1,11), padding=(0,5), groups=dim)
        # self.conv3_2 = nn.Conv2d(dim, dim, (11,1), padding=(5,0), groups=dim)

        # # 3 5 7 9 11 ------------------------------------------------------------------------------------------
        # self.conv0_1 = nn.Conv2d(dim, dim, (1,3), padding=(0,1), groups=dim)
        # self.conv0_2 = nn.Conv2d(dim, dim, (3,1), padding=(1,0), groups=dim)
        # self.conv1_1 = nn.Conv2d(dim, dim, (1,5), padding=(0,2), groups=dim)
        # self.conv1_2 = nn.Conv2d(dim, dim, (5,1), padding=(2,0), groups=dim)
        # self.conv2_1 = nn.Conv2d(dim, dim, (1,7), padding=(0,3), groups=dim)
        # self.conv2_2 = nn.Conv2d(dim, dim, (7,1), padding=(3,0), groups=dim)
        # self.conv3_1 = nn.Conv2d(dim, dim, (1,9), padding=(0,4), groups=dim)
        # self.conv3_2 = nn.Conv2d(dim, dim, (9,1), padding=(4,0), groups=dim)
        # self.conv4_1 = nn.Conv2d(dim, dim, (1,11), padding=(0,5), groups=dim)
        # self.conv4_2 = nn.Conv2d(dim, dim, (11,1), padding=(5,0), groups=dim)

        self.conv = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        u = x.clone()
        attn = self.conv0(x) 

        attn0 = self.conv0_1(attn) 
        attn0 = self.conv0_2(attn0) 

        attn1 = self.conv1_1(attn)
        attn1 = self.conv1_2(attn1)

        attn2 = self.conv2_1(attn)
        attn2 = self.conv2_2(attn2)

        # attn3 = self.conv3_1(attn)
        # attn3 = self.conv3_2(attn3)

        # attn4 = self.conv4_1(attn)
        # attn4 = self.conv4_2(attn4)

        attn = attn + attn0 + attn1 + attn2 # + attn3 + attn4
        
        attn = self.conv(attn)

        return attn * u

class DilatedAttentionModule(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)

        # 3 7 11 (base) ----------------------------------------------------------------------------------        
        self.conv0_1 = nn.Conv2d(dim, dim, (1, 3), padding=(0,1), groups=dim, dilation=(3,1))
        self.conv0_2 = nn.Conv2d(dim, dim, (3, 1), padding=(1,0), groups=dim, dilation=(1,3))
        self.conv1_1 = nn.Conv2d(dim, dim, (1, 7), padding=(0,3), groups=dim, dilation=(7,1))
        self.conv1_2 = nn.Conv2d(dim, dim, (7, 1), padding=(3,0), groups=dim, dilation=(1,7))
        self.conv2_1 = nn.Conv2d(dim, dim, (1, 11), padding=(0,5), groups=dim, dilation=(11,1))
        self.conv2_2 = nn.Conv2d(dim, dim, (11, 1), padding=(5,0), groups=dim, dilation=(1,11))

        # 3 5 7 ------------------------------------------------------------------------------------------
        # self.conv0_1 = nn.Conv2d(dim, dim, (1,3), padding=(0,1), groups=dim, dilation=(3,1))
        # self.conv0_2 = nn.Conv2d(dim, dim, (3,1), padding=(1,0), groups=dim, dilation=(1,3))
        # self.conv1_1 = nn.Conv2d(dim, dim, (1,5), padding=(0,2), groups=dim, dilation=(5,1))
        # self.conv1_2 = nn.Conv2d(dim, dim, (5,1), padding=(2,0), groups=dim, dilation=(1,5))
        # self.conv2_1 = nn.Conv2d(dim, dim, (1,7), padding=(0,3), groups=dim, dilation=(7,1))
        # self.conv2_2 = nn.Conv2d(dim, dim, (7,1), padding=(3,0), groups=dim, dilation=(1,7))

        # 5 7 9 ------------------------------------------------------------------------------------------
        # self.conv0_1 = nn.Conv2d(dim, dim, (1,5), padding=(0,2), groups=dim, dilation=(5,1))
        # self.conv0_2 = nn.Conv2d(dim, dim, (5,1), padding=(2,0), groups=dim, dilation=(1,5))
        # self.conv1_1 = nn.Conv2d(dim, dim, (1,7), padding=(0,3), groups=dim, dilation=(7,1))
        # self.conv1_2 = nn.Conv2d(dim, dim, (7,1), padding=(3,0), groups=dim, dilation=(1,7))
        # self.conv2_1 = nn.Conv2d(dim, dim, (1,9), padding=(0,4), groups=dim, dilation=(9,1))
        # self.conv2_2 = nn.Conv2d(dim, dim, (9,1), padding=(4,0), groups=dim, dilation=(1,9))

        # 7 9 11 ------------------------------------------------------------------------------------------
        # self.conv0_1 = nn.Conv2d(dim, dim, (1,7), padding=(0,3), groups=dim, dilation=(7,1))
        # self.conv0_2 = nn.Conv2d(dim, dim, (7,1), padding=(3,0), groups=dim, dilation=(1,7))
        # self.conv1_1 = nn.Conv2d(dim, dim, (1,9), padding=(0,4), groups=dim, dilation=(9,1))
        # self.conv1_2 = nn.Conv2d(dim, dim, (9,1), padding=(4,0), groups=dim, dilation=(1,9))
        # self.conv2_1 = nn.Conv2d(dim, dim, (1,11), padding=(0,5), groups=dim, dilation=(11,1))
        # self.conv2_2 = nn.Conv2d(dim, dim, (11,1), padding=(5,0), groups=dim, dilation=(1,11))


        # 3 5 7 9 ------------------------------------------------------------------------------------------
        # self.conv0_1 = nn.Conv2d(dim, dim, (1,3), padding=(0,1), groups=dim, dilation=(3,1))
        # self.conv0_2 = nn.Conv2d(dim, dim, (3,1), padding=(1,0), groups=dim, dilation=(1,3))
        # self.conv1_1 = nn.Conv2d(dim, dim, (1,5), padding=(0,2), groups=dim, dilation=(5,1))
        # self.conv1_2 = nn.Conv2d(dim, dim, (5,1), padding=(2,0), groups=dim, dilation=(1,5))
        # self.conv2_1 = nn.Conv2d(dim, dim, (1,7), padding=(0,3), groups=dim, dilation=(7,1))
        # self.conv2_2 = nn.Conv2d(dim, dim, (7,1), padding=(3,0), groups=dim, dilation=(1,7))
        # self.conv3_1 = nn.Conv2d(dim, dim, (1,9), padding=(0,4), groups=dim, dilation=(9,1))
        # self.conv3_2 = nn.Conv2d(dim, dim, (9,1), padding=(4,0), groups=dim, dilation=(1,9))

        # # 5 7 9 11 ------------------------------------------------------------------------------------------
        # self.conv0_1 = nn.Conv2d(dim, dim, (1,5), padding=(0,2), groups=dim, dilation=(5,1))
        # self.conv0_2 = nn.Conv2d(dim, dim, (5,1), padding=(2,0), groups=dim, dilation=(1,5))
        # self.conv1_1 = nn.Conv2d(dim, dim, (1,7), padding=(0,3), groups=dim, dilation=(7,1))
        # self.conv1_2 = nn.Conv2d(dim, dim, (7,1), padding=(3,0), groups=dim, dilation=(1,7))
        # self.conv2_1 = nn.Conv2d(dim, dim, (1,9), padding=(0,4), groups=dim, dilation=(9,1))
        # self.conv2_2 = nn.Conv2d(dim, dim, (9,1), padding=(4,0), groups=dim, dilation=(1,9))
        # self.conv3_1 = nn.Conv2d(dim, dim, (1,11), padding=(0,5), groups=dim, dilation=(11,1))
        # self.conv3_2 = nn.Conv2d(dim, dim, (11,1), padding=(5,0), groups=dim, dilation=(1,11))
        

        # # 3 5 7 9 11 ------------------------------------------------------------------------------------------
        # self.conv0_1 = nn.Conv2d(dim, dim, (1,3), padding=(0,1), groups=dim, dilation=(3,1))
        # self.conv0_2 = nn.Conv2d(dim, dim, (3,1), padding=(1,0), groups=dim, dilation=(1,3))
        # self.conv1_1 = nn.Conv2d(dim, dim, (1,5), padding=(0,2), groups=dim, dilation=(5,1))
        # self.conv1_2 = nn.Conv2d(dim, dim, (5,1), padding=(2,0), groups=dim, dilation=(1,5))
        # self.conv2_1 = nn.Conv2d(dim, dim, (1,7), padding=(0,3), groups=dim, dilation=(7,1))
        # self.conv2_2 = nn.Conv2d(dim, dim, (7,1), padding=(3,0), groups=dim, dilation=(1,7))
        # self.conv3_1 = nn.Conv2d(dim, dim, (1,9), padding=(0,4), groups=dim, dilation=(9,1))
        # self.conv3_2 = nn.Conv2d(dim, dim, (9,1), padding=(4,0), groups=dim, dilation=(1,9))
        # self.conv4_1 = nn.Conv2d(dim, dim, (1,11), padding=(0,5), groups=dim, dilation=(11,1))
        # self.conv4_2 = nn.Conv2d(dim, dim, (11,1), padding=(5,0), groups=dim, dilation=(1,11))

        self.conv = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        u = x.clone()
        attn = self.conv0(x) # 256 64 128

        attn0 = self.conv0_1(attn) # 256 64 127
        attn0 = self.conv0_2(attn0) # 256 63 127

        attn1 = self.conv1_1(attn)
        attn1 = self.conv1_2(attn1)

        attn2 = self.conv2_1(attn)
        attn2 = self.conv2_2(attn2)

        # attn3 = self.conv3_1(attn)
        # attn3 = self.conv3_2(attn3)

        # attn4 = self.conv4_1(attn)
        # attn4 = self.conv4_2(attn4)

        attn = attn + attn0 + attn1 + attn2 # + attn3 + attn4

        attn = self.conv(attn)

        return attn * u

class SpatialAttention(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        self.proj_1 = nn.Conv2d(d_model, d_model, 1)
        self.activation = nn.GELU()
        self.spatial_gating_unit = AttentionModule(d_model)
        # self.spatial_gating_unit = DilatedAttentionModule(d_model)
        self.proj_2 = nn.Conv2d(d_model, d_model, 1)

    def forward(self, x):
        shorcut = x.clone()
        x = self.proj_1(x)
        x = self.activation(x)
        x = self.spatial_gating_unit(x)
        x = self.proj_2(x)
        x = x + shorcut
        return x


        

