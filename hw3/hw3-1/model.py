import torch.nn as nn
import torch.nn.functional as F


class GeneratorNet(nn.Module):
    def __init__(self, input_size, init_size = 16):
        super(GeneratorNet, self).__init__()

        self.input_size = input_size
        self.init_size = init_size

        self.fc = nn.Linear(input_size, 128*init_size**2)
        self.conv_block = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2), # 32*32
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.Upsample(scale_factor=2), # 64*64
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1),
            nn.Tanh()
        )

    def forward(self, input):
        batch_size = input.size(0)
        x = self.fc(input)
        x = x.view(batch_size, 128, self.init_size, self.init_size)
        img = self.conv_block(x)
        return img

class DiscriminatorNet(nn.Module):
    def __init__(self, use_WGAN = False):
        self.use_WGAN = use_WGAN
        super(DiscriminatorNet, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1), # 32*32
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), # 16*16
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),# 8*8
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),# 4*4
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        self.fc = nn.Linear(4*4*256,1)
    def forward(self, image):
        batch_size = image.size(0)
        out = self.conv_block(image)
        out = out.view(batch_size, -1)
        if self.use_WGAN:
            validity = self.fc(out)
        else:
            validity = F.sigmoid(self.fc(out))
        return validity