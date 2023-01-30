import torch
import torch.nn as nn
from torchvision.models import VGG


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


# "A": vgg_11, "B": vgg_13, "D": vgg_16, "E": vgg_19
cfgs = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


# 提取VGG模型训练参数
class Extractor(nn.Module):
    def __init__(self, pretrained=False):
        super(Extractor, self).__init__()
        vgg16_bn = VGG(make_layers(cfgs["D"], batch_norm=True))
        if pretrained:
            vgg16_bn.load_state_dict(torch.load('./pths/vgg16_bn-6c64b313.pth'))
        self.features = vgg16_bn.features

    def forward(self, x):
        out = []
        for m in self.features:
            x = m(x)
            # 提取maxpool层为后续合并
            if isinstance(m, nn.MaxPool2d):
                out.append(x)
        return out[1:]


class UpSampling(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpSampling, self).__init__()

        self.layer = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, 3, 2, 1, 1)
        )

    def forward(self, x):
        return self.layer(x)


class Conv_BN_ReLU(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Conv_BN_ReLU, self).__init__()
        self.con_bn_relu = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.con_bn_relu(x)


class FPN_16(nn.Module):
    def __init__(self):
        super(FPN_16, self).__init__()
        self.extractor = Extractor()
        self.bn = nn.BatchNorm2d(512)
        self.upsampling1 = UpSampling(512, 256)
        self.conv1 = Conv_BN_ReLU(768, 128)
        self.upsampling2 = UpSampling(128, 64)
        self.conv2 = Conv_BN_ReLU(320, 64)
        self.upsampling3 = UpSampling(64, 32)
        self.conv3 = Conv_BN_ReLU(160, 32)

        self.conv4 = nn.Conv2d(32, 32, 3, 1, 1)

    def forward(self, inputs):
        f4, f3, f2, f1 = self.extractor(inputs)
        h1 = self.bn(f1)
        h2 = self.conv1(torch.cat((self.upsampling1(h1), f2), 1))
        h3 = self.conv2(torch.cat((self.upsampling2(h2), f3), 1))
        h4 = self.conv3(torch.cat((self.upsampling3(h3), f4), 1))

        out = self.conv4(h4)
        return out


class EAST(nn.Module):
    def __init__(self):
        super(EAST, self).__init__()
        self.fpn = FPN_16()
        self.inside_score = nn.Conv2d(32, 1, 1)
        self.side_vertex_code = nn.Conv2d(32, 2, 1)
        self.side_vertex_geo = nn.Conv2d(32, 4, 1)

    def forward(self, input):
        fpn = self.fpn(input)
        ins = self.inside_score(fpn)
        svc = self.side_vertex_code(fpn)
        svg = self.side_vertex_geo(fpn)

        out = torch.cat((ins, svc, svg), 1)
        return out


if __name__ == "__main__":
    model = EAST().cuda()
    x = torch.randn(2, 3, 224, 224).cuda()
    out = model(x)
    print(out.shape)
