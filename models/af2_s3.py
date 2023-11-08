import torch
from torch import nn

class SEAttention(nn.Module):
    def __init__(self, channel=512, reduction=16):
        super().__init__()
        self.reduction = reduction
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c) # sparse global pooling squeeze 
        y = self.fc(y).view(b, c, 1, 1) 
        return x * y.expand_as(x)


class AttentiveFeatureFusion(nn.Module):
    def __init__(self, in_dim, mid_dim, out_dim):
        super(AttentiveFeatureFusion, self).__init__()

        self.conv1 = nn.Conv2d(in_dim, mid_dim, 3, 1, 1)        
        self.conv12 = nn.Conv2d(mid_dim, out_dim, 5, 1, 2)
        self.conv13 = nn.Conv2d(out_dim, out_dim, 9, 1, 4)

        self.conv2 = nn.Conv2d(in_dim, mid_dim, 5, 1, 2)
        self.conv22 = nn.Conv2d(mid_dim, out_dim, 5, 1, 2)

        self.conv3 = nn.Conv2d(in_dim, out_dim, 13, 1, 6)

        self.conv4 = nn.Conv2d(out_dim, 1, 3, 1, 1)
        self.conv42 = nn.Conv2d(out_dim, 1, 3, 1, 1)
        self.conv43 = nn.Conv2d(out_dim, 1, 3, 1, 1)

        self.conv5 = nn.Conv2d(3, out_dim, 3, 1, 1)
        self.bn5 = nn.BatchNorm2d(out_dim)
        self.sigmoid = nn.Sigmoid()

        self.conv6 = nn.Conv2d(out_dim, out_dim, 3, 1, 1)

    def forward(self, input):
        x1 = self.conv1(input)
        x1 = self.conv12(x1)
        x1 = self.conv13(x1)

        x2 = self.conv2(input)
        x2 = self.conv22(x2)
        
        x3 = self.conv3(input)

        h1 = self.conv4(x1)
        h2 = self.conv42(x2)
        h3 = self.conv43(x3)

        f_enc = torch.cat([h1, h2, h3], dim=1)
        f_enc = torch.softmax(f_enc, dim=1)
        
        attn_residuals = self.conv5(f_enc)
        attn_residuals = self.bn5(attn_residuals)
        attn_residuals = self.sigmoid(attn_residuals)

        weight_matrix = torch.rand([x1.shape[1], f_enc.shape[1]])
        f_enc = nn.functional.linear(f_enc.view(1, 3, -1).transpose(1, 2), weight_matrix).view(1, x1.shape[1], 256, 1248)

        x1 = x1 * f_enc
        x2 = x2 * f_enc
        x3 = x3 * f_enc

        out = x1 + x2 + x3 + attn_residuals 
        out = self.conv6(out)

        return out, x1, x2, x3

class AdaptiveFeatureSelection(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(AdaptiveFeatureSelection, self).__init__()

        self.conv1 = nn.Conv2d(in_dim, in_dim, 3, 1, 1)
        self.conv2 = nn.Conv2d(in_dim, in_dim, 3, 1, 1)
        self.conv3 = nn.Conv2d(in_dim, in_dim, 3, 1, 1)

        self.vote = SEAttention(channel=out_dim, reduction=3)
        self.conv1x1 = nn.Conv2d(out_dim, in_dim, 1, 1)
    

    def forward(self, x1, x2, x3):

        x1_2 = self.conv1(x1)
        x1 = x1 + x1_2

        x2_2 = self.conv2(x2)
        x2 = x2 + x2_2

        x3_2 = self.conv3(x3)
        x3 = x3 + x3_2

        f_dec = torch.cat([x1, x2, x3], dim=1)
        # f_dec = self.relu(f_dec)
        se = self.vote(f_dec)
        theta = 0.35
        
        f_dec1 = f_dec * se * theta
        f_dec2 = f_dec * (1-theta)

        out = self.conv1x1(f_dec1 + f_dec2)

        return out


if __name__ == "__main__":
    AF = AttentiveFeatureFusion(3, 64, 32)
    
    import torch
    input = torch.rand([1, 3, 256, 1248])
    af, x1, x2, x3 = AF(input)

    AF2 = AdaptiveFeatureSelection(32, 96)

    af2 = AF2(x1, x2, x3)
    print()

