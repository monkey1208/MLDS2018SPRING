import torch
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor
from torch.autograd import Variable
import torch.autograd as autograd
import numpy as np
import argparse
import cv2
from tqdm import tqdm
from model import GeneratorNet, DiscriminatorNet

def set_requires_grad(net, switch):
    for param in net.parameters():
        param.requires_grad = switch
    return

def to_var(x, requires_grad=True):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, requires_grad= requires_grad)

def dcgan_train(data_loader, generator, discriminator, optimizer_G, optimizer_D, criterion, epoch):
    Tensor = torch.FloatTensor
    for e in range(epoch):
        desc = 'Epoch {}'.format(e)
        for i, im in enumerate(tqdm(data_loader, desc= desc)):
            optimizer_G.zero_grad()
            optimizer_D.zero_grad()
            batch_size = im[0].size(0)
            valid_label = to_var(Tensor(batch_size, 1).fill_(1.0), requires_grad=False)
            fake_label = to_var(Tensor(batch_size, 1).fill_(0.0), requires_grad=False)

            # Create Fake Image
            set_requires_grad(discriminator, False)
            noise = to_var(Tensor(np.random.normal(0, 1, (batch_size, opts.latent_size))))
            im_fake = generator(noise)
            g_loss = criterion(discriminator(im_fake), valid_label)
            #print('g_loss = {}'.format(g_loss.data))
            g_loss.backward()
            optimizer_G.step()

            # Discriminate Valid Image
            set_requires_grad(discriminator, True)
            im_valid = to_var(im[0])
            valid_loss = criterion(discriminator(im_valid), valid_label)
            #print('valid_loss = {}'.format(valid_loss))
            fake_loss = criterion(discriminator(im_fake.detach()), fake_label)
            #print('fake_loss = {}'.format(fake_loss))
            d_loss = (valid_loss+fake_loss)/2
            print('d_loss = {}'.format(d_loss.data))
            d_loss.backward()
            optimizer_D.step()
            im_fake_output = np.clip(im_fake.cpu().data.numpy(), 0, 1)
            if i % 40  == 0:
                im_fake_output = np.round(im_fake_output[0]*255)
                im_fake_output = np.transpose(im_fake_output , (1, 2, 0))
                print(im_fake_output.shape)
                cv2.imwrite('iter{}.jpg'.format(i), im_fake_output)

def compute_gradient_penalty(D, real_samples, fake_samples):
    # Random weight term for interpolation between real and fake samples
    Tensor = torch.FloatTensor
    if torch.cuda.is_available():
        Tensor = torch.cuda.FloatTensor
    alpha = Tensor(np.random.random((real_samples.size(0), 1, 1, 1)))
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = D(interpolates)
    fake = Variable(Tensor(real_samples.shape[0], 1).fill_(1.0), requires_grad=False)
    # Get gradient w.r.t. interpolates
    gradients = autograd.grad(outputs=d_interpolates, inputs=interpolates,
                              grad_outputs=fake, create_graph=True, retain_graph=True,
                              only_inputs=True)[0]
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

def wgan_train(data_loader, generator, discriminator, optimizer_G, optimizer_D, epoch, clip_val=0.01, LAMBDA=0.1, use_GP=False):
    Tensor = torch.FloatTensor
    for e in range(epoch):
        desc = 'Epoch {}'.format(e)
        for i, im in enumerate(tqdm(data_loader, desc= desc)):
            optimizer_G.zero_grad()
            optimizer_D.zero_grad()
            batch_size = im[0].size(0)
            im_valid = to_var(im[0])

            noise = to_var(Tensor(np.random.normal(0, 1, (batch_size, opts.latent_size))))
            set_requires_grad(discriminator, False)
            im_fake = generator(noise)
            g_loss = -torch.mean(discriminator(im_fake))
            g_loss.backward()
            optimizer_G.step()

            set_requires_grad(discriminator, True)
            if use_GP:
                gradient_penalty = compute_gradient_penalty(discriminator, im_valid.data, im_fake.data)
                d_loss = -torch.mean(discriminator(im_valid)) + torch.mean(discriminator(im_fake.detach())) + LAMBDA * gradient_penalty
            else:
                d_loss = -torch.mean(discriminator(im_valid)) + torch.mean(discriminator(im_fake.detach()))
            d_loss.backward()
            optimizer_D.step()
            #Weight Clipping
            if not use_GP:
                for param in discriminator.parameters():
                    param.data.clamp_(-clip_val, clip_val)
            im_fake_output = np.clip(im_fake.cpu().data.numpy(), 0, 1)
            if i % 40  == 0:
                im_fake_output = np.round(im_fake_output[0]*255)
                im_fake_output = np.transpose(im_fake_output , (1, 2, 0))
                cv2.imwrite('iter{}.jpg'.format(i), im_fake_output)

def main(opts):
    im_folder = ImageFolder(root=opts.input, transform= ToTensor())
    data_loader = torch.utils.data.DataLoader(im_folder,
                                              batch_size=opts.batch_size,
                                              shuffle=True)

    USE_WGAN = opts.use_wgan

    discriminator = DiscriminatorNet(use_WGAN=USE_WGAN)
    generator = GeneratorNet(input_size=100)
    criterion_dcgan = torch.nn.BCELoss()

    if torch.cuda.is_available():
        print('Using CUDA')
        discriminator.cuda()
        generator.cuda()
    optimizer_G = torch.optim.Adam(generator.parameters(), lr= 0.0002)
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr= 0.0002)

    if opts.use_wgan or opts.use_wgan_gp:
        print('WGAN Training')
        wgan_train(data_loader,generator,discriminator,optimizer_G,optimizer_D, opts.epoch, use_GP=opts.use_wgan_gp)
    else:
        print('DCGAN Training')
        dcgan_train(data_loader,generator,discriminator,optimizer_G,optimizer_D, criterion_dcgan, opts.epoch)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', type=str, default= 'resized_data', dest='input', help='Animate Face Input File Path')
    parser.add_argument('-b', type=int, default= 128, dest='batch_size', help='Batch Number')
    parser.add_argument('-l', type=int, default= 100, dest='latent_size', help='Latent Number')
    parser.add_argument('-e', type=int, default= 1, dest='epoch', help='Epoch Number')
    parser.add_argument('-WGAN', action= 'store_true', default= False, dest= 'use_wgan', help='use WGAN Weight Clipping')
    parser.add_argument('-WGAN-GP', action= 'store_true', default= False, dest= 'use_wgan_gp', help='use WGAN Weight Clipping')
    opts = parser.parse_args()
    main(opts)
