
import torch
from torch import nn


class SC_UNET_Encoder(nn.Module):
    def __init__(self, dim=48):
        super(SC_UNET_Encoder, self).__init__()
        self.conv_block1 = self.conv_block(1, dim)
        self.conv_block2 = self.conv_block(dim, dim*2)
        self.conv_block3 = self.conv_block(dim*2, dim*4)
        self.conv_block4 = self.conv_block(dim*4, dim*8)
        self.conv_block5 = self.conv_block(dim*8, dim*16)

        self.maxpool = nn.MaxPool2d(2)

    def conv_block(self, in_channel, out_channels):
        layer = nn.Sequential(
            nn.Conv2d(in_channel, out_channels, 3, 1, padding=1),
            nn.ELU(),
            nn.BatchNorm2d(out_channels),
            nn.Conv2d(out_channels, out_channels, 3, 1, padding=1),
            nn.ELU(),
            nn.BatchNorm2d(out_channels)
        )
        return layer


    def forward(self, x):
        f1 = self.conv_block1(x) # 48 112 592
        f1 = self.maxpool(f1) # 48 56 296

        f2 = self.conv_block2(f1)
        f2 = self.maxpool(f2)

        f3_1= self.conv_block3(f2)
        f3_2 = self.maxpool(f3_1)

        f4_1 = self.conv_block4(f3_2)
        f4_2 = self.maxpool(f4_1)

        f5 = self.conv_block5(f4_2)

        return f5, f4_1, f3_1
    

class SC_UNET_Decoder(nn.Module):
    def __init__(self, dim=768):
        super(SC_UNET_Decoder, self).__init__()
        self.up1 = self.deconv(dim, dim/2)
        self.conv_block1 = self.conv_block(dim, dim/2)

        self.up2 = self.deconv(dim/2, dim/2/2)
        self.conv_block2 = self.conv_block(dim/2, dim/2/2)
        
        self.up3 = self.deconv(dim/2/2, dim/2/2/2)
        self.conv_block3 = self.conv_block(dim/2/2/2, dim/2/2/2)
        
        self.up4 = self.deconv(dim/2/2/2, dim/2/2/2/2)
        self.conv_block4 = self.conv_block(dim/2/2/2/2, dim/2/2/2/2)

        self.conv = nn.Conv2d(int(dim/2/2/2/2), 3, 1, 1)

    def deconv(self, in_channels, out_channels):
        in_channels, out_channels = int(in_channels), int(out_channels)
        return nn.ConvTranspose2d(in_channels, out_channels, 4, 2, padding=1)

    def conv_block(self, in_channels, out_channels):
        in_channels, out_channels = int(in_channels), int(out_channels)
        layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, padding=1),
            nn.ELU(),
            nn.BatchNorm2d(out_channels),
            nn.Conv2d(out_channels, out_channels, 3, 1, padding=1),
            nn.ELU(),
            nn.BatchNorm2d(out_channels)
        )
        return layer

    def forward(self, f5, f4_1, f3_1):
        f5 = self.up1(f5)
        up1 = torch.cat((f5, f4_1), dim=1)  # 16n 
        up1 = self.conv_block1(up1)         # 8n

        up2 = self.up2(up1)
        up2 = torch.cat((up2, f3_1), dim=1) # 8n 
        up2 = self.conv_block2(up2)         # 4n 192

        up3 = self.up3(up2)                 # 2n 
        up3 = self.conv_block3(up3)         # 2n

        up4 = self.up4(up3)                 # n 
        up4 = self.conv_block4(up4)         # n 

        rgb = self.conv(up4)                # 3
        rgb = nn.sigmoid(rgb)

        return rgb 
    

if __name__ == "__main__":
    input = torch.rand([1, 1, 112, 592])
    encoder = SC_UNET_Encoder(dim=48)
    decoder = SC_UNET_Decoder(dim=768)

    f5, f4_1, f3_1 = encoder(input)
    logits = decoder(f5, f4_1, f3_1)
    
    from torchinfo import summary
    print(summary(encoder, (1, 1, 112, 592)))

    from torchviz import make_dot
    make_dot(encoder(input.cuda())).render('sc_unet_test', format='png')
