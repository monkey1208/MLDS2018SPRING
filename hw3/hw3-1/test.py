
import matplotlib
matplotlib.use('Agg')
import torch
from torch.autograd import Variable
from model import GeneratorNet, DiscriminatorNet
import numpy as np
import cv2
import ipdb
import matplotlib.pyplot as plt
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--generator', type=str, default= 'generator', dest='generator', help='generator path')
parser.add_argument('--output', type=str, default= './', help='output')
args = parser.parse_args()
g_path = args.generator
def save_imgs(generator):
    r, c = 5, 5
    noise = np.random.normal(0, 1, (r * c, 100))
    # gen_imgs should be shape (25, 64, 64, 3)
    gen_imgs = generator.predict(noise)
    fig, axs = plt.subplots(r, c)
    cnt = 0
    for i in range(r):
        for j in range(c):
            axs[i,j].imshow(gen_imgs[cnt, :,:,:])
            axs[i,j].axis('off')
            cnt += 1
    fig.savefig("output.png")
    plt.close()
def to_var(x, requires_grad=True):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, requires_grad= requires_grad)
generator = GeneratorNet(input_size=100)
discriminator = DiscriminatorNet(use_WGAN=False)
if torch.cuda.is_available():
    generator = generator.cuda()
generator.load_state_dict(torch.load(g_path))

Tensor = torch.FloatTensor
#noise = np.random.normal(0,1,(100,100))
#np.save('vector.npy', noise)
#noise = to_var(Tensor(np.random.normal(0, 1, (100, 100))))
noise = np.load('model/vector.npy')
noise = to_var(Tensor(noise))
im_fake = generator(noise)
im_fake_output = np.clip(im_fake.cpu().data.numpy(), 0, 1)
losses = []
gen_imgs = []
for i in range(100):
    im_fake_output0 = im_fake_output[i]
    im_fake_output0 = np.transpose(im_fake_output0 , (1, 2, 0))
    #cv2.imwrite('test{}.jpg'.format(i), im_fake_output0*255)
    gen_imgs.append(im_fake_output0) 
gen_imgs = np.array(gen_imgs)
r,c = 5,5
cnt = 0
fig, axs = plt.subplots(r, c)
for i in range(r):
    for j in range(c):
        axs[i,j].imshow(gen_imgs[cnt, :,:,:])
        axs[i,j].axis('off')
        cnt += 1
fig.savefig("gan.png")
plt.close()
