import torch
from torch import Tensor
from torch import nn


'''
https://github.com/hei4/mnist-segmentation-detection
'''
class SimpleUNet(nn.Module):
    def __init__(self, in_channels: int =1, num_classes: int =11) -> None:
        super().__init__()

        self.encoder = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, 64, 3, stride=1, padding=1),     # [N, 64, 56, 56]
                nn.ReLU(inplace=True),

                nn.Conv2d(64, 64, 3, stride=1, padding=1, bias=False),  # [N, 64, 56, 56]
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True)
            ),
            nn.Sequential(
                nn.MaxPool2d(2),    # [N, 64, 14, 14]

                nn.Conv2d(64, 128, 3, stride=1, padding=1, bias=False),     # [N, 128, 28, 28]
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                
                nn.Conv2d(128, 128, 3, stride=1, padding=1, bias=False),    # [N, 128, 28, 28]
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True)
            )
        ])

        self.bottle_neck = nn.Sequential(
            nn.MaxPool2d(2),    # [N, 128, 7, 7]

            nn.Conv2d(128, 256, 3, stride=1, padding=1, bias=False),    # [N, 256, 14, 14]
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 256, 3, stride=1, padding=1, bias=False),    # [N, 256, 14, 14]
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)   # [N, 128, 28, 28]
        )

        self.decoder = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(128+256, 128, 3, stride=1, padding=1, bias=False),    # [N, 128, 28, 28]
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),

                nn.Conv2d(128, 128, 3, stride=1, padding=1, bias=False),    # [N, 128, 28, 28]
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),

                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)   # [N, 128, 56, 56]
            ),

            nn.Sequential(
                nn.Conv2d(64+128, 64, 3, stride=1, padding=1, bias=False),  # [N, 64, 56, 56]
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),

                nn.Conv2d(64, 64, 3, stride=1, padding=1, bias=False),  # [N, 64, 56, 56]
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                
                nn.Conv2d(64, num_classes, 1, stride=1, padding=0)     # [N, 11, 56, 56]
            )
        ])

    def forward(self, x: Tensor) -> Tensor:
        h_list = []
        for block in self.encoder:
            x = block(x)
            h_list.append(x)

        x = self.bottle_neck(x)

        for h, block in zip(h_list[::-1], self.decoder):
            x = block(torch.cat([h, x], dim=1))

        return x

    def calculate_loss(self, inputs, labels):
        loss = nn.CrossEntropyLoss()
        return loss(self(inputs), labels)