import torch
import torch.nn as nn
from torch.nn import Module


class ResidualBlock(Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        padding=1,
        stride=1,
        drop_ratio=0,
    ):
        super(ResidualBlock, self).__init__()

        self.expand = True if stride == 2 else False
        self.drop_ratio = drop_ratio

        if self.expand:
            self.conv1 = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                padding=padding,
                stride=2,
            )
        else:
            self.conv1 = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                padding=padding,
                stride=1,
            )

        self.conv2 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding,
            stride=1,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.act1 = nn.ReLU()
        self.act2 = nn.ReLU()
        self.drop1 = nn.Dropout(p=0.3)
        self.drop2 = nn.Dropout(p=0.3)

        if self.expand:
            self.skip_connection = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                padding=0,
                stride=2,
            )
        else:
            self.skip_connection = nn.Identity()

    def forward(self, x):
        z = self.conv1(x)
        z = self.bn1(z)
        z = self.act1(z)

        if self.drop_ratio != 0:
            z = self.drop1(z)

        z = self.conv2(z)
        z = self.bn2(z)

        x = self.skip_connection(x)
        x = x + z
        x = self.act2(x)

        if self.drop_ratio != 0:
            x = self.drop2(x)

        return x


class ResNet(Module):
    def __init__(
        self, in_channels, start_channels, class_num, blocks=[2, 2, 2, 2], drop_ratio=0
    ):
        super(ResNet, self).__init__()

        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=start_channels,
            kernel_size=7,
            padding=3,
            stride=2,
        )
        self.bn = nn.BatchNorm2d(start_channels)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.act = nn.ReLU()

        self.layer1 = nn.Sequential(
            *[
                ResidualBlock(start_channels, start_channels, drop_ratio=drop_ratio)
                for i in range(blocks[0])
            ]
        )
        self.layer2 = nn.Sequential(
            *[
                ResidualBlock(
                    start_channels, start_channels * 2, stride=2, drop_ratio=drop_ratio
                )
                if i == 0
                else ResidualBlock(
                    start_channels * 2, start_channels * 2, drop_ratio=drop_ratio
                )
                for i in range(blocks[1])
            ]
        )
        self.layer3 = nn.Sequential(
            *[
                ResidualBlock(
                    start_channels * 2,
                    start_channels * 2 * 2,
                    stride=2,
                    drop_ratio=drop_ratio,
                )
                if i == 0
                else ResidualBlock(
                    start_channels * 2 * 2,
                    start_channels * 2 * 2,
                    drop_ratio=drop_ratio,
                )
                for i in range(blocks[2])
            ]
        )
        self.layer4 = nn.Sequential(
            *[
                ResidualBlock(
                    start_channels * 2 * 2,
                    start_channels * 2 * 2 * 2,
                    stride=2,
                    drop_ratio=drop_ratio,
                )
                if i == 0
                else ResidualBlock(
                    start_channels * 2 * 2 * 2,
                    start_channels * 2 * 2 * 2,
                    drop_ratio=drop_ratio,
                )
                for i in range(blocks[3])
            ]
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten(start_dim=1)
        self.fc = nn.Linear(
            in_features=start_channels * 2 * 2 * 2, out_features=class_num
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.maxpool(x)
        x = self.act(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.fc(x)

        return x


# test code
if __name__ == "__main__":
    model = ResNet(3, 64, 5, drop_ratio=0)

    img = torch.randn(64, 3, 320, 320)
    x = model(img)
    print(x.shape)

    print(sum([p.numel() for p in model.parameters() if p.requires_grad]))
