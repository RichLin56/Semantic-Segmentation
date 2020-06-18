import torch
import torch.nn as nn
import torchvision


def load_resnet(name: str):
    backbones = ["resnet50", "resnet101", "resnet152"]
    assert (name in backbones), '{0} does not exist in {1}'.format(name, backbones)
    if name == "resnet50":
        return torchvision.models.resnet.resnet50(pretrained=True)
    elif name == "resnet101":
        return torchvision.models.resnet.resnet101(pretrained=True)
    elif name == "resnet152":
        return torchvision.models.resnet.resnet101(pretrained=True)
        

class ConvBlock(nn.Module):
    """
    Helper module that consists of a Conv -> BN -> ReLU
    """

    def __init__(self, in_channels, out_channels, padding=1, kernel_size=3, stride=1, with_nonlinearity=True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, padding=padding, kernel_size=kernel_size, stride=stride)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.with_nonlinearity = with_nonlinearity

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.with_nonlinearity:
            x = self.relu(x)
        return x


class Bridge(nn.Module):
    """
    This is the middle layer of the UNet which just consists of some
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.bridge = nn.Sequential(
            ConvBlock(in_channels, out_channels),
            ConvBlock(out_channels, out_channels)
        )

    def forward(self, x):
        return self.bridge(x)


class UpBlockForUNetWithResNet(nn.Module):
    """
    Up block that encapsulates one up-sampling step which consists of Upsample -> ConvBlock -> ConvBlock
    """

    def __init__(self, in_channels, out_channels, up_conv_in_channels=None, up_conv_out_channels=None,
                 upsampling_method="conv_transpose"):
        super().__init__()

        if up_conv_in_channels == None:
            up_conv_in_channels = in_channels
        if up_conv_out_channels == None:
            up_conv_out_channels = out_channels
        #TODO: Bug with bilinear upsampling
        if upsampling_method == "conv_transpose":
            self.upsample = nn.ConvTranspose2d(up_conv_in_channels, up_conv_out_channels, kernel_size=2, stride=2)
        elif upsampling_method == "bilinear":
            self.upsample = nn.Sequential(
                nn.Upsample(mode='bilinear', scale_factor=2),
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)
            )
        self.conv_block_1 = ConvBlock(in_channels, out_channels)
        self.conv_block_2 = ConvBlock(out_channels, out_channels)

    def forward(self, up_x, down_x):
        """
        :param up_x: this is the output from the previous up block
        :param down_x: this is the output from the down block
        :return: upsampled feature map
        """
        x = self.upsample(up_x)
        x = torch.cat([x, down_x], 1)
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        return x


class UResNet(nn.Module):

    def __init__(self, in_channels, out_channels, backbone="resnet50", upsampling_method="conv_transpose"):
        super().__init__()
        resnet = load_resnet(backbone)
        down_blocks = []
        up_blocks = []
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.depth = 6   
        self.upsampling_method = upsampling_method    
        self.input_block = nn.Sequential(*list(resnet.children()))[:3]

        if self.in_channels == 1:
            self.input_block[0] = nn.Conv2d(self.in_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
            self.input_block[0].weight = torch.nn.Parameter(self.input_block[0].weight[:, 0:1, :, :])

        self.input_pool = list(resnet.children())[3]
        for bottleneck in list(resnet.children()):
            if isinstance(bottleneck, nn.Sequential):
                down_blocks.append(bottleneck)
        self.down_blocks = nn.ModuleList(down_blocks)
        self.bridge = Bridge(2048, 2048)
        up_blocks.append(UpBlockForUNetWithResNet(2048, 1024, upsampling_method=self.upsampling_method))
        up_blocks.append(UpBlockForUNetWithResNet(1024, 512, upsampling_method=self.upsampling_method))
        up_blocks.append(UpBlockForUNetWithResNet(512, 256, upsampling_method=self.upsampling_method))
        up_blocks.append(UpBlockForUNetWithResNet(in_channels=128 + 64, out_channels=128,
                                                    up_conv_in_channels=256, up_conv_out_channels=128,
                                                    upsampling_method=self.upsampling_method))
        up_blocks.append(UpBlockForUNetWithResNet(in_channels=64 + self.in_channels, out_channels=64,
                                                    up_conv_in_channels=128, up_conv_out_channels=64,
                                                    upsampling_method=self.upsampling_method))

        self.up_blocks = nn.ModuleList(up_blocks)

        self.out = nn.Conv2d(64, self.out_channels, kernel_size=1, stride=1)

    def forward(self, x, with_output_feature_map=False):
        pre_pools = dict()
        pre_pools[f"layer_0"] = x
        x = self.input_block(x)
        pre_pools[f"layer_1"] = x
        x = self.input_pool(x)

        for i, block in enumerate(self.down_blocks, 2):
            x = block(x)
            if i == (self.depth - 1):
                continue
            pre_pools[f"layer_{i}"] = x

        x = self.bridge(x)

        for i, block in enumerate(self.up_blocks, 1):
            key = f"layer_{self.depth - 1 - i}"
            x = block(x, pre_pools[key])
        output_feature_map = x
        x = self.out(x)
        del pre_pools
        if with_output_feature_map:
            return x, output_feature_map
        else:
            return x

if __name__ == "__main__":
    model = UResNet(1, 1, backbone="resnet101", upsampling_method="conv_transpose")
    ins = torch.ones((1, 1, 224, 224))
    outs = model(ins)
