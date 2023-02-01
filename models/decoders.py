from modules import Concatenate

import torch
import torch.nn as nn
import torchvision.transforms.functional as TransFunc 


class DUpsampling(nn.Module):
    def __init__(self, inplanes, scale, num_class=21, pad=0):
        super(DUpsampling, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, num_class * scale * scale, kernel_size=1, padding = pad,bias=False)
        self.scale = scale
    
    def forward(self, x):
        x = self.conv1(x)
        N, C, H, W = x.size()

        # N, H, W, C
        x_permuted = x.permute(0, 2, 3, 1) 

        # N, H, W*scale, C/scale
        x_permuted = x_permuted.contiguous().view((N, H, W * self.scale, int(C / (self.scale))))

        # N, W*scale,H, C/scale
        x_permuted = x_permuted.permute(0, 2, 1, 3)
        # N, W*scale,H*scale, C/(scale**2)
        x_permuted = x_permuted.contiguous().view((N, W * self.scale, H * self.scale, int(C / (self.scale * self.scale))))

        # N,C/(scale**2),W*scale,H*scale
        x = x_permuted.permute(0, 3, 2, 1)
        
        return x

class Deconv_TransUnet_Decorder(nn.Module):
    def __init__(self,in_filters, out_filters, *args):
        super(Deconv_TransUnet_Decorder, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_filters, out_channels=out_filters, kernel_size=(3, 3), padding='same')
        self.batch_norm = nn.BatchNorm2d(num_features = out_filters)
        self.Relu_activate = nn.ReLU()
        self.upsampling = nn.UpsamplingBilinear2d(scale_factor=2)
        self.concat = Concatenate(axis=1)
        self.Dropout = nn.Dropout(0.3)
    def forward(self, x, sk_connect):
        x = self.conv(x)  
        x = self.batch_norm(x)
        x = self.Dropout(x)
        x = self.Relu_activate(x)              
        x = self.upsampling(x)
        x = self.concat(x, sk_connect)
        return x

class TransUNet_Decorder(nn.Module): # resnest 50 
    def __init__(self, fusion_lev, *args):
        super(TransUNet_Decorder, self).__init__()
        self.fusion_lev = fusion_lev

        self.deconv_1 = Deconv_TransUnet_Decorder(in_filters=1024, out_filters=512)
        self.deconv_2 = Deconv_TransUnet_Decorder(in_filters=512*2, out_filters=256)
        self.deconv_3 = Deconv_TransUnet_Decorder(in_filters=256*2, out_filters=64) 
        self.conv_4 = nn.Conv2d(in_channels=64*2, out_channels=64, kernel_size=(3, 3), padding='same')

        self.dupsample = DUpsampling(inplanes=64, scale=2, num_class=64)
        self.batch_norm = nn.BatchNorm2d(num_features=64)
        self.Relu_activate = nn.ReLU()
        self.upsampling = nn.UpsamplingBilinear2d(scale_factor=2)
        self.conv_5 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding='same')

        self.Dropout = nn.Dropout(0.3)
                                      
    def forward(self, f1, f2, f3, f4):
        x = self.deconv_1(f4, f3)   
        x = self.deconv_2(x, f2)          
        x = self.deconv_3(x, f1)         
        x = self.conv_4(x)          
        x = self.batch_norm(x)      
        x = self.Dropout(x)         
        x = self.Relu_activate(x)   

        # x = self.dupsample(x)
        x = self.upsampling(x)      
        x = self.conv_5(x)          
        x = self.batch_norm(x)      
        x = self.Dropout(x)         
        x = self.Relu_activate(x)   

        # x = self.upsampling(x)      
        # x = self.conv_5(x)          
        # x = self.batch_norm_5(x)    
        # x = self.Dropout(x)      
        # x = self.Relu_activate_5(x)

        return x

class TransUNet_101(nn.Module):
    def __init__(self, fusion_lev, *args):
        super(TransUNet_101, self).__init__()

        self.fusion_lev = fusion_lev

        self.deconv_1 = Deconv_TransUnet_Decorder(in_filters=1024, out_filters=512)
        self.deconv_2 = Deconv_TransUnet_Decorder(in_filters=512*2, out_filters=256)
        self.deconv_3 = Deconv_TransUnet_Decorder(in_filters=256*2, out_filters=128) 
        self.conv_4 = nn.Conv2d(in_channels=128*2, out_channels=64, kernel_size=(3, 3), padding='same')

        self.dupsample = DUpsampling(inplanes=64, scale=2, num_class=64)
        self.batch_norm = nn.BatchNorm2d(num_features=64)
        self.Relu_activate = nn.ReLU()
        self.upsampling = nn.UpsamplingBilinear2d(scale_factor=2)
        self.conv_5 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding='same')

        self.Dropout = nn.Dropout(0.3)

    def forward(self, f1, f2, f3, f4):
        x = self.deconv_1(f4, f3)        
        x = self.deconv_2(x, f2)   
        x = self.deconv_3(x, f1)      
        x = self.conv_4(x)        
        x = self.batch_norm(x)  
        x = self.Dropout(x)        
        x = self.Relu_activate(x) 

        # x = self.dupsample(x)
        x = self.upsampling(x)     
        x = self.conv_5(x)          
        x = self.batch_norm(x)     
        x = self.Dropout(x)                
        x = self.Relu_activate(x)  

        # x = self.upsampling(x)     
        # x = self.conv_5(x)          
        # x = self.batch_norm_5(x)   
        # x = self.Dropout(x)              
        # x = self.Relu_activate_5(x) 

        return x



class Decoder_FCN(nn.Module):# FCN의 naive upsampling 디코더 
    """
    Parameters:
    layer_channels (List): number of input & output channels of the each conv_up blocks
    """
    def __init__(self, layer_channels, include_top:bool=False, n_class:int=0):
        super(Decoder_FCN, self).__init__()

        self.layers_channels = layer_channels

        self.up_trans = nn.ModuleList(
            [nn.ConvTranspose2d(layer_channel, layer_channel_n, kernel_size=2, stride=2) for layer_channel, layer_channel_n in zip(self.layers_channels[::-1][:-2], self.layers_channels[::-1][1:-1])])
       
        self.double_conv_ups = nn.ModuleList(
            [self.__double_conv(layer_channel, layer_channel//2) for layer_channel in self.layers_channels[::-1][:-2]])   

        self.include_top=include_top
        self.n_class=n_class 
        if self.include_top:       
            self.final_conv = nn.Conv2d(self.layers_channels[::-1][-2], n_class, kernel_size=1)

    def __double_conv(self, in_channels, out_channels):
        conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        return conv
        
    def forward(self, x):                
        concat_layers = concat_layers[::-1] # 리스트의 순서를 역순으로 뒤집음 [1024, 512, 256, 128, 64, in_channels]

        for up_trans, double_conv_up  in zip(self.up_trans, self.double_conv_ups):            
            x = double_conv_up(x)#conv
            x = up_trans(x)#size * 2

        if self.include_top and self.n_class != 0:
            x = self.final_conv(x)
        else:
            x = x

        return x

class Decoder_MLA(nn.Module):# SETR 논문의 decoder 구조 모방   
    """
    Parameters:
    layer_channels (List): number of input & output channels of the each conv_up blocks
    """
    def __init__(self, layer_channels, include_top:bool=False, n_class:int=0):
        super(Decoder_UNET, self).__init__()

        self.layers_channels = layer_channels # 1, 1/2, 1/4, 1/8, 1/16, 1/32

        self.up_trans = nn.ModuleList(
            [nn.ConvTranspose2d(layer_channel, layer_channel_n, kernel_size=2, stride=2) for layer_channel, layer_channel_n in zip(self.layers_channels[::-1][:-2], self.layers_channels[::-1][1:-1])])
       
        self.double_conv_ups = nn.ModuleList(
            [self.__double_conv(layer_channel, layer_channel//2) for layer_channel in self.layers_channels[::-1][:-2]])   
        
        self.include_top=include_top
        self.n_class=n_class 
        if self.include_top:       
            self.final_conv = nn.Conv2d(self.layers_channels[::-1][-2], n_class, kernel_size=1)

    def __double_conv(self, in_channels, out_channels):
        conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        return conv
        
    def forward(self, x, concat_layers, add=True):     
        
        concat_layers = concat_layers[::-1] # 역순

        for k in range(0,3): # 1/32 -> 1/16 -> 1/8 -> 1/4
            x=self.up_trans[k](x)

        c1=self.up_trans[1](concat_layers[0]) # 1/16 -> 1/8
        c1=self.up_trans[2](c1) # 1/8 -> 1/4
        if add:
            c1=torch.add(x,c1,dim=1)
        else:
            c1 = torch.cat((x,c1), axis=1)

        c2=self.up_trans[2](concat_layers[1]) # 1/8 -> 1/4
        if add:
            c2 = torch.add(c1, c2, dim=1)
            c3 = torch.add(c2, concat_layers[2]) # 1/4
        else:
            c2 = torch.cat((c1, concat_layers[2]), axis=1)
            c3 = torch.cat((c2, concat_layers[2]), axis=1)

        x = self.__double_conv(self.layers_channels[::-1][3], self.layers_channels[::-1][3])(x)
        c1 = self.__double_conv(self.layers_channels[::-1][3], self.layers_channels[::-1][3])(c1)
        c2 = self.__double_conv(self.layers_channels[::-1][3], self.layers_channels[::-1][3])(c2)
        c3 = self.__double_conv(self.layers_channels[::-1][3], self.layers_channels[::-1][3])(c3)

        for i in range(2):#1/4 -> 1/2 -> 1
            x = self.up_trans[i+3](x)
            c1 = self.up_trans[i+3](c1)
            c2 = self.up_trans[i+3](c2)
            c3 = self.up_trans[i+3](c3)
        x = torch.cat((x, c1, c2, c3), dim=1)
        x = self.__double_conv(self.layers_channels[::-1][-1], self.layers_channels[::-1][-1])(x)

        if self.include_top and self.n_class != 0:
            x = self.final_conv(x)
        else:
            x = x

        return x

class Decoder_UNET(nn.Module):
    """
    Parameters:
    layer_channels (List): number of input & output channels of the each conv_up blocks
    """
    def __init__(self, layer_channels, include_top:bool=False, n_class:int=0):
        super(Decoder_UNET, self).__init__()

        self.layers_channels = layer_channels

        self.up_trans = nn.ModuleList(
            [nn.ConvTranspose2d(layer_channel, layer_channel_n, kernel_size=2, stride=2) for layer_channel, layer_channel_n in zip(self.layers_channels[::-1][:-2], self.layers_channels[::-1][1:-1])])
       
        self.double_conv_ups = nn.ModuleList(
            [self.__double_conv(layer_channel, layer_channel//2) for layer_channel in self.layers_channels[::-1][:-2]])   
        
        self.include_top=include_top
        self.n_class=n_class 
        if self.include_top:       
            self.final_conv = nn.Conv2d(self.layers_channels[::-1][-2], n_class, kernel_size=1)

    def __double_conv(self, in_channels, out_channels):
        conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        return conv
        
    def forward(self, x, concat_layers):        
        
        concat_layers = concat_layers[::-1]#리스트의 순서를 역순으로 뒤집음

        for up_trans, double_conv_up, concat_layer  in zip(self.up_trans, self.double_conv_ups, concat_layers):
            x = up_trans(x)
            if x.shape != concat_layer.shape:
                x = TransFunc.resize(x, concat_layer.shape[2:])            
            concatenated = torch.cat((concat_layer, x), dim=1)
            x = double_conv_up(concatenated)

        if self.include_top and self.n_class != 0:
            x = self.final_conv(x)
        else:
            x = x

        return x

class MLP_SegFormer(nn.Module): # for fusion 
    def __init__(self, num_cls, hidden_dim, embed_dim, fusion_lev):
        super(MLP_SegFormer, self).__init__()
        self.num_cls = num_cls
        self.hidden_dim = hidden_dim
        self.embed_dim = embed_dim
        self.fusion_lev = fusion_lev

        # conv layer
        self.conv1 = {"img" : nn.Conv2d(in_channels=self.embed_dim[0], out_channels=self.hidden_dim, kernel_size=1), "rem" : nn.Conv2d(in_channels=self.embed_dim[0], out_channels=self.hidden_dim, kernel_size=1)}
        self.conv2 = {"img" : nn.Conv2d(in_channels=self.embed_dim[1], out_channels=self.hidden_dim, kernel_size=1), "rem" : nn.Conv2d(in_channels=self.embed_dim[1], out_channels=self.hidden_dim, kernel_size=1)}
        self.conv3 = nn.Conv2d(in_channels=self.embed_dim[2], out_channels=self.hidden_dim, kernel_size=1) 
        self.conv4 = nn.Conv2d(in_channels=self.embed_dim[3], out_channels=self.hidden_dim, kernel_size=1) 
        self.Convolution = nn.Conv2d
        self.Convolution2 = nn.Conv2d(in_channels=self.hidden_dim, out_channels=self.num_cls, kernel_size=1)
        # Upsample layer
        self.Upsample1 = {"img" : nn.Upsample(scale_factor=4, mode='bilinear'), "rem" : nn.Upsample(scale_factor=4, mode='bilinear')}
        self.Upsample2 = {"img" : nn.Upsample(scale_factor=8, mode='bilinear'), "rem" : nn.Upsample(scale_factor=8, mode='bilinear')}
        self.Upsample3 = nn.Upsample(scale_factor=16, mode='bilinear')
        self.Upsample4 = nn.Upsample(scale_factor=32, mode='bilinear')

        # concat layer
        self.concat = torch.cat

    def forward(self, f1_img, f1_rem, f2_img, f2_rem, f3, f4):
        x_list = list()
        x_list.append(self.Upsample1["img"](self.conv1["img"](f1_img)))
        x_list.append(self.Upsample1["rem"](self.conv1["rem"](f1_rem)))
        x_list.append(self.Upsample2["img"](self.conv2["img"](f2_img)))
        x_list.append(self.Upsample2["rem"](self.conv2["rem"](f2_rem)))
        x_list.append(self.Upsample3(self.conv3(f3)))
        x_list.append(self.Upsample4(self.conv4(f4)))
        x = self.concat(x_list,dim=1)
        x = self.Convolution(in_channels=(len(x_list)*self.hidden_dim), out_channels=self.hidden_dim, kernel_size=1)(x)
        return x

class U_MLA(nn.Module):# SETR 논문의 decoder 구조 모방   
    """
    Parameters:
    layer_channels (List): number of input & output channels of the each conv_up blocks
    """
    def __init__(self, layer_channels, include_top:bool=False, n_class:int=0):
        super(U_MLA, self).__init__()

        self.layers_channels = layer_channels

        self.up_trans = nn.ModuleList(
            [nn.ConvTranspose2d(layer_channel, layer_channel_n, kernel_size=2, stride=2) for layer_channel, layer_channel_n in zip(self.layers_channels[::-1][:-2], self.layers_channels[::-1][1:-1])])

        self.double_conv_ups = nn.ModuleList(
            [self.__double_conv(layer_channel*2, layer_channel) for layer_channel in self.layers_channels[::-1][1:-1]]) 

        self.up_trans_final = nn.ModuleList(
            [nn.ConvTranspose2d(layer_channel, layer_channel_n, kernel_size=2, stride=2) for layer_channel, layer_channel_n in zip(self.layers_channels[::-1][1:-2], self.layers_channels[::-1][2:-1])])
                       
        self.include_top=include_top
        self.n_class=n_class 
        if self.include_top:       
            self.final_conv = nn.Conv2d(self.layers_channels[::-1][-2]*4, n_class, kernel_size=1)

    def __double_conv(self, in_channels, out_channels):
        conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        return conv
        
    def forward(self, x, concat_layers):        
        
        concat_layers = concat_layers[::-1]#리스트의 순서를 역순으로 뒤집음
        layers = []
        for up_trans, double_conv_up, concat_layer  in zip(self.up_trans, self.double_conv_ups, concat_layers):
            x = up_trans(x)#크기 2배 & 다음 특징맵과 동일한 수의 채널로 조정
            if x.shape != concat_layer.shape:
                x = TransFunc.resize(x, concat_layer.shape[2:])            
            concatenated = torch.cat((concat_layer, x), dim=1)
            x = double_conv_up(concatenated)
            layers.append(x) # 1/8, 1/4, 1/2, 1 
        
        final_layers=[x]
        for i, l in enumerate(layers[::-1][1:]):#1, 1/2, 1/4, 1/8
            for k in range(i,-1,-1):#1회, 2회, 3회
                l = self.up_trans_final[len(layers)-k](l) #   
            final_layers.append(l) 
        x = torch.cat(final_layers, dim=1)
        if self.include_top and self.n_class != 0:
            x = self.final_conv(x)
        else:
            x = x

        return x

" -------------------------------------------------------------------------------------- "
class MLA_add(nn.Module):
    def __init__(self, n_class, include_top=False):
        super(MLA_add, self).__init__()
        self.layers_channels = [3, 64, 256, 512, 1024] # 1, 1/2, 1/4, 1/8, 1/16, 1/32

        self.up_trans = nn.ModuleList(
            # [nn.ConvTranspose2d(layer_channel, layer_channel_n, kernel_size=2, stride=2) for layer_channel, layer_channel_n in zip(self.layers_channels[::-1][:-2], self.layers_channels[::-1][1:-1])])
            [nn.ConvTranspose2d(layer_channel, layer_channel_n, kernel_size=2, stride=2) for layer_channel, layer_channel_n in zip(self.layers_channels[::-1][:-1], self.layers_channels[::-1][1:])])
       
        self.double_conv_ups = nn.ModuleList(
            [self.__double_conv(layer_channel, layer_channel//2) for layer_channel in self.layers_channels[::-1][:-2]])   
        
        self.include_top=include_top
        self.n_class=n_class 
        if self.include_top:       
            self.final_conv = nn.Conv2d(self.layers_channels[::-1][-2], self.n_class, kernel_size=1)

        
    def __double_conv(self, in_channels, out_channels):
        conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        # ).cuda()
        return conv

    
    def forward(self, f1, f2, f3, f4):
        #       dim     w   h       
        #   f1  64      48  160     1/2
        #   f2  256     24  80      1/4    
        #   f3  512     12  40      1/8
        #   f4  1024    6   20      1/16

        for i in range(0,3):
            f4 = self.up_trans[i](f4) # 64  48  160
        
        for i in range(1,3):
            f3 = self.up_trans[i](f3) # 64  48  160
        
        add1 = torch.add(f3, f4)      # 64  48  160
        
        f2 = self.up_trans[2](f2)     # 64  48  160
        add2 = torch.add(add1, f2)    # 64  48  160

        add3 = torch.add(add2, f1)    # 64  48  160

        f4 = self.__double_conv(self.layers_channels[::-1][3], self.layers_channels[::-1][3])(f4)       # 64  48  160
        add1 = self.__double_conv(self.layers_channels[::-1][3], self.layers_channels[::-1][3])(add1)   # 64  48  160
        add2 = self.__double_conv(self.layers_channels[::-1][3], self.layers_channels[::-1][3])(add2)   # 64  48  160
        add3 = self.__double_conv(self.layers_channels[::-1][3], self.layers_channels[::-1][3])(add3)   # 64  48  160

        f4 = self.up_trans[-1](f4)
        add1 = self.up_trans[-1](add1)
        add2 = self.up_trans[-1](add2)
        add3 = self.up_trans[-1](add3)
         
        x = torch.cat((f4, add1, add2, add3), dim=1)
        x = self.__double_conv(x.shape[1], 64)(x)

        return  x # 64 96 320

class MLA_cat(nn.Module):
    def __init__(self, n_class, include_top=False):
        super(MLA_cat, self).__init__()
        self.layers_channels = [3, 64, 256, 512, 1024] # 1, 1/2, 1/4, 1/8, 1/16, 1/32

        self.up_trans = nn.ModuleList(
            # [nn.ConvTranspose2d(layer_channel, layer_channel_n, kernel_size=2, stride=2) for layer_channel, layer_channel_n in zip(self.layers_channels[::-1][:-2], self.layers_channels[::-1][1:-1])])
            [nn.ConvTranspose2d(layer_channel, layer_channel_n, kernel_size=2, stride=2) for layer_channel, layer_channel_n in zip(self.layers_channels[::-1][:-1], self.layers_channels[::-1][1:])])
       
        self.double_conv_ups = nn.ModuleList(
            [self.__double_conv(layer_channel, layer_channel//2) for layer_channel in self.layers_channels[::-1][:-2]])   
        
        self.include_top=include_top
        self.n_class=n_class 
        if self.include_top:       
            self.final_conv = nn.Conv2d(self.layers_channels[::-1][-2], self.n_class, kernel_size=1)
        self.reduction = nn.Conv2d(128, 64, 1)

    def __double_conv(self, in_channels, out_channels):
        conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        # )
        ).cuda()
        return conv
    
    def forward(self, f1, f2, f3, f4):
        #       dim     w   h       
        #   f1  64      48  160     1/2
        #   f2  256     24  80      1/4    
        #   f3  512     12  40      1/8
        #   f4  1024    6   20      1/16

        for i in range(0,3):
            f4 = self.up_trans[i](f4)           # 64  48  160
        
        for i in range(1,3):
            f3 = self.up_trans[i](f3)           # 64  48  160
        
        add1 = torch.cat((f3, f4), dim=1)       # 128  48  160
        add1 = self.reduction(add1)
        
        f2 = self.up_trans[2](f2)               # 64  48  160
        add2 = torch.cat((add1, f2), dim=1)              # 64  48  160
        add2 = self.reduction(add2)

        add3 = torch.cat((add2, f1), dim=1)              # 64  48  160
        add3 = self.reduction(add3)

        f4 = self.__double_conv(self.layers_channels[::-1][3], self.layers_channels[::-1][3])(f4)       # 64  48  160
        add1 = self.__double_conv(self.layers_channels[::-1][3], self.layers_channels[::-1][3])(add1)   # 64  48  160
        add2 = self.__double_conv(self.layers_channels[::-1][3], self.layers_channels[::-1][3])(add2)   # 64  48  160
        add3 = self.__double_conv(self.layers_channels[::-1][3], self.layers_channels[::-1][3])(add3)   # 64  48  160

        f4 = self.up_trans[-1](f4)
        add1 = self.up_trans[-1](add1)
        add2 = self.up_trans[-1](add2)
        add3 = self.up_trans[-1](add3)
         
        x = torch.cat((f4, add1, add2, add3), dim=1)
        x = self.__double_conv(x.shape[1], 64)(x)

        return  x # 64 96 320

class PUP(nn.Module):
    def __init__(self):
        super(PUP, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=3, padding='same')
        self.batch_norm1 = nn.BatchNorm2d(num_features=512)
        self.conv2 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3, padding='same')
        self.batch_norm2 = nn.BatchNorm2d(num_features=256)
        self.conv3 = nn.Conv2d(in_channels=256, out_channels=64, kernel_size=3, padding='same')
        self.batch_norm34 = nn.BatchNorm2d(num_features=64)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding='same')
        
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)
        self.Dropout = nn.Dropout(0.3)
        self.Relu_activate = nn.ReLU()

    def forward(self, f4):     
        x = self.conv1(f4)     
        x = self.batch_norm1(x)
        x = self.Dropout(x)
        x = self.Relu_activate(x)
        x = self.upsample(x)   

        x = self.conv2(x)      
        x = self.batch_norm2(x)
        x = self.Dropout(x)
        x = self.Relu_activate(x)
        x = self.upsample(x)   

        x = self.conv3(x)      
        x = self.batch_norm34(x)
        x = self.Dropout(x)
        x = self.Relu_activate(x)        
        x = self.upsample(x)   

        x = self.conv4(x)      
        x = self.batch_norm34(x)
        x = self.Dropout(x)
        x = self.Relu_activate(x)
        x = self.upsample(x)   

        return x

class UNet_MLA_cat(nn.Module):
    def __init__(self, n_class, include_top=False):
        super(UNet_MLA_cat, self).__init__()
        self.layers_channels = [3, 64, 256, 512, 1024] # 1, 1/2, 1/4, 1/8, 1/16, 1/32

        self.up_trans = nn.ModuleList(
            # [nn.ConvTranspose2d(layer_channel, layer_channel_n, kernel_size=2, stride=2) for layer_channel, layer_channel_n in zip(self.layers_channels[::-1][:-2], self.layers_channels[::-1][1:-1])])
            [nn.ConvTranspose2d(layer_channel, layer_channel_n, kernel_size=2, stride=2) for layer_channel, layer_channel_n in zip(self.layers_channels[::-1][:-1], self.layers_channels[::-1][1:])])
       
        self.double_conv_ups = nn.ModuleList(
            [self.__double_conv(layer_channel, layer_channel//2) for layer_channel in self.layers_channels[::-1][:-2]])   
        
        self.include_top=include_top
        self.n_class=n_class 
        if self.include_top:       
            self.final_conv = nn.Conv2d(self.layers_channels[::-1][-2], self.n_class, kernel_size=1)
        self.reduction = nn.Conv2d(128, 64, 1)
        self.reductions = nn.ModuleList(
            [nn.Conv2d(c*2, c, 1) for c in self.layers_channels[::-1][:-1]]
        )

    def __double_conv(self, in_channels, out_channels):
        conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        # ).cuda()
        return conv
    
    def forward(self, f1, f2, f3, f4):
        #       dim     w   h       
        #   f1  64      48  160     1/2
        #   f2  256     24  80      1/4    
        #   f3  512     12  40      1/8
        #   f4  1024    6   20      1/16
        # ---------------- UNet ----------------
        f4 = self.up_trans[0](f4)               #   512     12  40
        cat1 = torch.cat((f3, f4), dim=1)    
        cat1 = self.reductions[1](cat1)

        cat2 = self.up_trans[1](cat1)
        cat2 = torch.cat((f2, cat2), dim=1)
        cat2 = self.reductions[2](cat2)         #   256 24  80

        cat3 = self.up_trans[2](cat2)
        cat3 = torch.cat((f1, cat3), dim=1)
        cat3 = self.reductions[3](cat3)         #   64  48  160
        
        # ---------------- MLA ----------------
        up_trans1 = self.up_trans[1](f4)
        up_trans1 = self.up_trans[2](up_trans1)

        up_trans2 = self.up_trans[1](cat1)
        up_trans2 = self.up_trans[2](up_trans2)
        mla_cat1 = torch.cat((up_trans1, up_trans2), dim=1)
        mla_cat1 = self.reduction(mla_cat1)

        up_trans3 = self.up_trans[2](cat2)
        mla_cat2 = torch.cat((up_trans3, mla_cat1), dim=1)
        mla_cat2 = self.reduction(mla_cat2)

        mla_cat3 = torch.cat((f1, mla_cat2), dim=1)
        mla_cat3 = self.reduction(mla_cat3)
        
        x4 = self.__double_conv(self.layers_channels[::-1][3], self.layers_channels[::-1][3])(mla_cat1)
        x3 = self.__double_conv(self.layers_channels[::-1][3], self.layers_channels[::-1][3])(mla_cat2)
        x2 = self.__double_conv(self.layers_channels[::-1][3], self.layers_channels[::-1][3])(mla_cat3)
        x1 = self.__double_conv(self.layers_channels[::-1][3], self.layers_channels[::-1][3])(f1)

        x4 = self.up_trans[-1](x4)
        x3 = self.up_trans[-1](x3)
        x2 = self.up_trans[-1](x2)
        x1 = self.up_trans[-1](x1)

        x = torch.cat((x4, x3, x2, x1), dim=1)
        x = self.__double_conv(x.shape[1], 64)(x)


        return x  # 64 96 320
" -------------------------------------------------------------------------------------- "
            

        

        

