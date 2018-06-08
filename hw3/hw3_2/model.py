import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import ipdb


class Generator(nn.Module):
    def __init__(self, tag_dim, embedded_dim=64, noise_dim=100, eye_dim=13 , hair_dim=16):
        super(Generator, self).__init__()
        self.dense = nn.Linear(noise_dim + embedded_dim, 128*16*16)
        self.noise_dim = noise_dim
        self.tag_dim = tag_dim
        self.eye_dim = eye_dim
        self.hair_dim = hair_dim
        self.embedded_dim = embedded_dim
        self.feat_dense = nn.Linear(tag_dim * embedded_dim, embedded_dim)
        self.eye_embedding = nn.Embedding(self.eye_dim, self.embedded_dim)
        self.hair_embedding = nn.Embedding(self.hair_dim, self.embedded_dim)
        self.embedding = nn.Embedding(self.tag_dim, self.embedded_dim)
        self.net = nn.Sequential(
            nn.Upsample(scale_factor=2),    # 64*128*32*32
            nn.Conv2d(128, 128, 3, 1, 1),   # 16 16
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.Conv2d(64, 3, 3, 1, 1),
            nn.Tanh()
        )

    def forward(self, inputs, tags):
        batch_size = tags.size(0)
        feat = self.embedding(tags).view(batch_size, -1)
        feat = F.relu(feat)
        latent = torch.cat((inputs, feat), 1)
        out = self.dense(latent)
        out = out.view(inputs.size(0), 128, 16, 16)
        out = self.net(out)
        return out


class Discriminator(nn.Module):
    def __init__(self, tag_dim, embedded_dim=64, out_channel=32, n_channel=3, eye_dim=13 , hair_dim=16):
        super(Discriminator, self).__init__()
        self.eye_dim=eye_dim
        self.hair_dim = hair_dim
        self.outch = out_channel
        self.tag_dim = tag_dim
        self.embedded_dim = embedded_dim
        self.n_channel = n_channel
        self.net = nn.Sequential(
            nn.Conv2d(self.n_channel, self.outch, 3, 2, 1),
            nn.BatchNorm2d(self.outch),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.outch, self.outch * 2, 3, 2, 1),
            nn.BatchNorm2d(self.outch * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.outch * 2, self.outch * 4, 3, 2, 1),
            nn.BatchNorm2d(self.outch * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.outch * 4, self.outch * 8, 3, 2, 1),
            nn.BatchNorm2d(self.outch * 8),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.embedding = nn.Embedding(self.tag_dim, self.embedded_dim)
        self.eye_embedding = nn.Embedding(self.eye_dim, self.embedded_dim)
        self.hair_embedding = nn.Embedding(self.hair_dim, self.embedded_dim)
        self.feat_dense = nn.Linear(tag_dim * embedded_dim, embedded_dim)
        self.dense = nn.Linear(4 * 4 * 128, 1)
        latent_dim = self.outch * 8 + self.embedded_dim
        self.net_2 = nn.Sequential(
            nn.Conv2d(latent_dim, self.outch * 4, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
        )



    def forward(self, image, tags):
        batch_size = image.size(0)
        feat = self.embedding(tags).view(batch_size, -1)
        feat = F.relu(feat)
        out = self.net(image)
        feat = feat.repeat(out.size(2), out.size(3), 1, 1).permute(2, 3, 0, 1)
        #ipdb.set_trace()
        out = torch.cat((out, feat), 1)
        out = self.net_2(out)
        out = out.view(batch_size, -1)
        out = self.dense(out)
        out = F.sigmoid(out)
        return out
