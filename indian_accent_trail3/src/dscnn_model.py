import torch
import torch.nn as nn

class DSCNN(nn.Module):
    def __init__(self, num_classes):
        super(DSCNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),  # Standard Convolution
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.depthwise_separable_convs = nn.Sequential(
            # Depthwise separable convolution block 1
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, groups=64),  # Depthwise
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=1, stride=1),  # Pointwise
            nn.BatchNorm2d(128),
            nn.ReLU(),
            
            # Depthwise separable convolution block 2
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, groups=128),  # Depthwise
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=1, stride=1),  # Pointwise
            nn.BatchNorm2d(256),
            nn.ReLU(),

            # Depthwise separable convolution block 3
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, groups=256),  # Depthwise
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=1, stride=1),  # Pointwise
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.depthwise_separable_convs(x)
        x = self.global_avg_pool(x)
        x = self.fc(x)
        return x
