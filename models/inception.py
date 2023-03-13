
import torch
import torch.nn as nn


class inceptionV1(nn.Module):
    def __init__(self, in_dim, out_dim1, mid_dim3, out_dim3, mid_dim5, out_dim5, pool):
        super(inceptionV1, self).__init__()
        self.split1 = self.conv1(in_dim, out_dim1)
        self.split2 = self.conv1_3(in_dim, mid_dim3, out_dim3)
        self.split3 = self.conv1_5(in_dim, mid_dim5, out_dim5)
        self.split4 = self.max1_conv1(in_dim, pool)
        
    def conv1(self, in_dim, out_dim):
        model = nn.Sequential(
            nn.Conv2d(in_dim, out_dim,1, 1),
            nn.ReLU()
        )
        return model
    
    def conv1_3(self, in_dim, mid_dim, out_dim):
        model = nn.Sequential(
            nn.Conv2d(in_dim, mid_dim, 1, 1),
            nn.ReLU(),
            nn.Conv2d(mid_dim, out_dim, 3, 1, 1),
            nn.ReLU()
        )        
        return model
    
    def conv1_5(self, in_dim, mid_dim, out_dim):
        model = nn.Sequential(
            nn.Conv2d(in_dim, mid_dim, 1, 1),
            nn.ReLU(),
            nn.Conv2d(mid_dim, out_dim, 5, 1, 2),
            nn.ReLU()
        )
        return model 

    def max1_conv1(self, in_dim, out_dim):
        model = nn.Sequential(
            nn.MaxPool2d(3, 1, 1),
            nn.Conv2d(in_dim, out_dim, 1, 1),
            nn.ReLU()
        )
        return model

    def forward(self, x):
        x1 = self.split1(x)
        x2 = self.split2(x)
        x3 = self.split3(x)
        x4 = self.split4(x)
        
        out = torch.cat([x1, x2, x3, x4], dim=1)
        
        return out 


class inception_A(nn.Module):
    def __init__(self, in_dim, out_dim1=96, mid_dim3=64, out_dim3=96, mid_dim4=64, mid2_dim4=96, out_dim4=96) -> None:
        super(inception_A, self).__init__()
        
        self.split1 = self.conv1(in_dim, out_dim1)
        self.split2 = self.pool_conv1(in_dim, out_dim1)
        self.split3 = self.conv1_3(in_dim, mid_dim3, out_dim3)
        self.split4 = self.conv1_3_3(in_dim, mid_dim4, mid2_dim4, out_dim4)
        
    def conv1(self, in_dim, out_dim):
        model = nn.Sequential(
             nn.Conv2d(in_dim, out_dim, 1, 1),
             nn.ReLU()
        )
        return model
    
    def pool_conv1(slef, in_dim, out_dim):
        model = nn.Sequential(
            nn.AvgPool2d(3,1,1),
            nn.Conv2d(in_dim, out_dim, 1,1),
            nn.ReLU()
        )
        return model 
    
    def conv1_3(self, in_dim, mid_dim, out_dim):
        
        model = nn.Sequential(
            nn.Conv2d(in_dim, mid_dim, 1, 1),
            nn.Conv2d(mid_dim, out_dim, 3, 1, 1)
        )
        return model 
        
    def conv1_3_3(self,in_dim, mid_dim, mid2_dim, out_dim):
        model = nn.Sequential(
            nn.Conv2d(in_dim, mid_dim, 1, 1),
            nn.ReLU(),
            nn.Conv2d(mid_dim, mid2_dim, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(mid2_dim, out_dim, 3, 1, 1),
            nn.ReLU()
        )
        
        return model
        
    def forward(self, x):
        x1 = self.split1(x)
        x2 = self.split2(x)
        x3 = self.split3(x)
        x4 = self.split4(x)
        
        out = torch.cat([x1, x2, x3, x4], 1)
        
        return out

class inception_B(nn.Module):
    def __init__(self, in_dim, out_dim1=384, pool=128, mid_dim3=192, mid2_dim3=224, out_dim3=256, mid_dim4=192, mid2_dim4=192, mid3_dim4=224, mid4_dim4=224, out_dim4=256, kernel_size=7) -> None:
        super(inception_B, self).__init__()
        self.split1 = self.conv1(in_dim, out_dim1)
        self.split2 = self.pool_conv1(in_dim, pool)
        self.split3 = self.conv1_n(in_dim, mid_dim3, mid2_dim3, out_dim3, kernel_size)
        self.split4 = self.conv1_n_n(in_dim, mid_dim4, mid2_dim4, mid3_dim4, mid4_dim4, out_dim4, kernel_size)
        
    def conv1(self, in_dim, out_dim):
        model = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, 1, 1),
            nn.ReLU()
        )
        return model
    
    def pool_conv1(self, in_dim, out_dim):
        model = nn.Sequential(
            nn.AvgPool2d(3, 1, 1),
            nn.Conv2d(in_dim, out_dim, 1, 1),
            nn.ReLU()
        )
        return model
    
    def conv1_n(self, in_dim, mid_dim, mid2_dim, out_dim, kernel_size):
        model = nn.Sequential(
            nn.Conv2d(in_dim, mid_dim, 1, 1),
            nn.ReLU(),
            nn.Conv2d(mid_dim, mid2_dim, (1,kernel_size), padding='same'),
            nn.Conv2d(mid2_dim, out_dim, (kernel_size,1), padding='same'),
            nn.ReLU()
        )
        return model 
        
    def conv1_n_n(self, in_dim, mid_dim, mid2_dim, mid3_dim, mid4_dim, out_dim, kernel_size):
        model = nn.Sequential(
            nn.Conv2d(in_dim, mid_dim, 1, 1),
            nn.ReLU(),
            nn.Conv2d(mid_dim, mid2_dim, (1,kernel_size), padding='same'),  
            nn.Conv2d(mid2_dim, mid3_dim, (kernel_size,1), padding='same'),
            nn.ReLU(),
            nn.Conv2d(mid3_dim, mid4_dim, (1,kernel_size), padding='same'),
            nn.Conv2d(mid4_dim, out_dim, (kernel_size,1), padding='same'),
            nn.ReLU()
        )
        return model 
    
    def forward(self, x):
        x1 = self.split1(x)
        x2 = self.split2(x)
        x3 = self.split3(x)
        x4 = self.split4(x)
        
        out = torch.concat([x1, x2, x3, x4], 1)
        
        return out 
    
class inception_C(nn.Module):
    def __init__(self, in_dim) -> None:
        super(inception_C, self).__init__()
        self.layer1 = self.conv1(in_dim, out_dim=256)
        self.layer2 = self.pool_conv1(in_dim, out_dim=256)
        self.layer3 = self.conv1(in_dim, out_dim=384)
        self.layer3_1 = self.conv_13(in_dim=384, out_dim=256)
        self.layer3_2 = self.conv_13(in_dim=384, out_dim=256)
        
        self.layer4 = self.conv1_3_3(in_dim, mid_dim=384, mid2_dim=448, out_dim=512)
        self.layer4_1 = self.conv_13(in_dim=512, out_dim=256)
        self.layer4_2 = self.conv_31(in_dim=512, out_dim=256)

    def conv1(self, in_dim, out_dim):
        model = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, 1, 1),
            nn.ReLU()
        )
        return model
    
    def pool_conv1(self, in_dim, out_dim):
        model = nn.Sequential(
            nn.AvgPool2d(3,1,1),
            self.conv1(in_dim, out_dim)            
        )
        return model
    
    def conv_13(self, in_dim, out_dim):
        model = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, (1,3), padding='same'),
            nn.ReLU())
        return model
        
    def conv_31(self, in_dim, out_dim):
        model = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, (3,1), padding='same'),
            nn.ReLU())
        return model
        
    def conv1_3(self, in_dim, mid_dim, out_dim):
        model = nn.Sequential(
            self.conv1(in_dim, mid_dim),
            self.conv3(mid_dim, out_dim)
        )
        return model
    
    def conv1_3_3(self, in_dim, mid_dim, mid2_dim, out_dim):
        model = nn.Sequential(
            self.conv1(in_dim, mid_dim),
            nn.Conv2d(mid_dim, mid2_dim, (1,3), padding='same'),
            nn.Conv2d(mid2_dim, out_dim, (3,1), padding='same'),
            nn.ReLU()
        )
        return model
    
    def forward(self, x):
        out1 = self.layer1(x)
        out2 = self.layer2(x)
        
        out3 = self.layer3(x)
        out3_1 = self.layer3_1(out3)
        out3_2 = self.layer3_2(out3)
        
        out4 = self.layer4(x)
        out4_1 = self.layer4_1(out4)
        out4_2 = self.layer4_2(out4)
        
        out = torch.concat([out1, out2, out3_1, out3_2, out4_1, out4_2], dim=1)
        
        return out 
    
if __name__ == "__main__":
    
    f = torch.randn([5, 3, 96, 320])
    inceptionv1 = inceptionV1(3, 64, 96, 128, 16, 32, 32)
    inceptionA = inception_A(in_dim=3)
    inceptionB = inception_B(in_dim=3)
    inceptionC = inception_C(in_dim=3)
    
    out1 = inceptionv1(f) # c=256
    out2 = inceptionA(f) # c=384
    out3 = inceptionB(f) # c=1024
    out4 = inceptionC(f) # c=1536
    
    print(out1.shape)
    print(out2.shape)
    print(out3.shape)
    print(out4.shape)
