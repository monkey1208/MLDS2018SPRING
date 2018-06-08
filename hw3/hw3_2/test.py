
import matplotlib
matplotlib.use('Agg')
import torch
from torch.autograd import Variable
from model import Generator, Discriminator
import numpy as np
import cv2
import matplotlib.pyplot as plt
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--tag', type=str, required=True)
parser.add_argument('--clean', type=str)
parser.add_argument('--extra',type=str)
parser.add_argument('--output')
parser.add_argument('-g', type=str, default= 'generator', dest='generator', help='generator path')
args = parser.parse_args()
g_path = args.generator
tag_dim = 131
def save_imgs(generator):
    r, c = 5, 5
    noise = np.random.normal(0, 1, (r * c, 100))
    np.save("noise.npy", noise)
    #noise =
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
generator = Generator(tag_dim)
discriminator = Discriminator(tag_dim)
if torch.cuda.is_available():
    generator = generator.cuda()
generator.load_state_dict(torch.load(args.generator))

f = open(args.clean, "r")
hair_voc = {}
eye_voc = {}
pair = {}
feat = {}
index = 0
hair_index = 0
eye_index = 0
color_hair = ['orange hair', 'white hair', 'aqua hair', 'gray hair', 'green hair', 'red hair', 'purple hair',
             'pink hair', 'blue hair', 'black hair', 'brown hair', 'blonde hair']
color_eye = ['gray eyes', 'black eyes', 'orange eyes', 'pink eyes', 'yellow eyes', 'aqua eyes', 'purple eyes',
             'green eyes', 'brown eyes', 'red eyes', 'blue eyes']
print("----------start creating feature tags----------")
while True:
    line = f.readline()
    if line == "":
        break
    need = 0
    pair_str = ""
    features = line.split(",")[1].strip(" ").strip("\n").split("\t")
    for feature in features:
        feature = feature.split(":")[0]
        if feature in color_eye:
            need += 1
            eyes = feature
        if feature in color_hair:
            hair = feature
            need += 1

    if need == 2:
        pair_str = hair + " " + eyes
    if pair_str not in pair and need == 2:
        pair[pair_str] = index
        index += 1
f.close()
f = open(args.extra, "r")
while True:
    line = f.readline()
    if line == "":
        break
    idx, features = line.split(",")
    features = features.strip("\n")
    if features not in pair:
        pair[features] = index
        index+=1
f.close()

f = open(args.tag, "r")
pair_np = np.zeros(len(pair))
flag = 0
while True:
    pairs = np.zeros(len(pair))
    line = f.readline()
    if line == "":
        break
    idx, features = line.split(",")
    feature = features.strip('\n')
    if flag == 0:
        pair_np[pair[feature]] = 1
        flag += 1
    else:
        pairs[pair[feature]] = 1
        pair_np = np.vstack((pair_np, pairs))

f.close()


test_tags_file = pair_np
tag_f = test_tags_file
tag_t = np.zeros((tag_f.shape[0]))
for j in range(tag_f.shape[0]):
    for k in range(tag_f.shape[1]):
        if tag_f[j][k] == 1:
            tag_t[j] = k
tag_t = to_var(torch.LongTensor(tag_t), requires_grad = False)
Tensor = torch.FloatTensor
#noise = torch.randn(25,100)
#noise = np.random.normal(0, 1, (25, 100))
noise = np.load("hw3_2/noise.npy")
#np.save("noise.npy", noise)
noise = to_var(Tensor(noise))

im_fake = generator(noise, tag_t)
im_fake_output = np.clip(im_fake.cpu().data.numpy(), 0, 1)* 127.5 + 127.5

losses = []
gen_imgs = []
for i in range(25):
    im_fake_output0 = im_fake_output[i]
    im_fake_output0 = np.transpose(im_fake_output0 , (1, 2, 0))
    #cv2.imwrite('test{}.jpg'.format(i), im_fake_output0)
    gen_imgs.append(im_fake_output0) 
gen_imgs = np.array(gen_imgs)
r,c = 5,5
cnt = 0
fig, axs = plt.subplots(r, c)
gen_imgs /= 256
for i in range(r):
    for j in range(c):
        axs[i,j].imshow(gen_imgs[cnt, :,:,:])
        axs[i,j].axis('off')
        cnt += 1
fig.savefig(args.output)
plt.close()
