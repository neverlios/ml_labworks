import imageio
import numpy as np
import torch
from torchvision import transforms as tfs
from torch.utils.data import Dataset, DataLoader
from os.path import join
import cv2
import torch.nn as nn
import torch.nn.functional as F
import time
import os
import wandb
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
def debayering(raw):
    channel_green_1 = raw[0::2, 1::2]
    channel_green_2 = raw[1::2, 0::2]
    channel_blue = raw[1::2, 1::2]
    channel_red = raw[0::2, 0::2]

    image = np.array((channel_red, channel_green_1, channel_green_2, channel_blue))# array instead of dstack
    image = image.astype(float) / 255.0 # norm 255 instead of 255 * 4
    return image
class DatasetZurich(Dataset):
    def __init__(self, dataset_dir, size, istest=False, transform=None):
        if istest:
            self.raw_dir = join(dataset_dir, 'test', 'huawei_raw')
            self.dslr_dir = join(dataset_dir, 'test', 'canon')
            
        else:
            self.raw_dir = join(dataset_dir, 'train', 'huawei_raw')
            self.dslr_dir = join(dataset_dir, 'train', 'canon')
        
        self.dataset_size = size
        self.istest = istest
        self.transform = transform
    
    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        raw = imageio.v3.imread(self.raw_dir + '/' + str(idx) + '.png')
        raw = np.float32(debayering(raw))
        raw = torch.from_numpy(raw) 
        
        dslr = imageio.v3.imread(self.dslr_dir + '/' + str(idx) + '.jpg').astype('uint8')
        # dslr = cv2.resize(dslr, (raw.shape[1], raw.shape[2]))
        dslr = torch.from_numpy(dslr.transpose((2,0,1))) / 255.0
        
        if self.transform:
            raw, dslr = self.transform(raw, dslr)
        return raw, dslr, str(idx)

class DatasetMai(Dataset):
    def __init__(self, dataset_dir, size, transform=None):
        self.dataset_size = size
        self.transform = transform 
        self.dataset_dir = dataset_dir

    def __len__(self):
         return self.dataset_size
    
    def __getitem__(self, index):
        raw = imageio.v3.imread(self.dataset_dir + '/mediatek_raw/' + str(index) + '.png')
        raw = np.float32(debayering(raw))
        raw = torch.from_numpy(raw) 
        
        dslr = imageio.v3.imread(self.dataset_dir + '/fujifilm/' + str(index) + '.png').astype('uint8')
        # dslr = cv2.resize(dslr, (raw.shape[1], raw.shape[2]))
        dslr = torch.from_numpy(dslr.transpose((2,0,1))) / 255.0
        
        if self.transform:
            raw, dslr = self.transform(raw, dslr)
        return raw, dslr, str(index)
class Encoder(nn.Module):
    def __init__(self, ):
        super().__init__()
        self.conv1 = nn.Conv2d( 4, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1)     
        self.conv5 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1)
        self.flat = nn.Flatten() 
        self.linear1 = nn.Linear(64*64, 512)
        self.norm = nn.BatchNorm1d(512)
        self.linear2 = nn.Linear(512, 256)
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = self.flat(x)
        embending = F.relu(self.linear1(x))
        embending = self.norm(embending)
        embending = F.relu(self.linear2(embending))
        embending = nn.functional.normalize(embending, p=2, dim=1)
        return x, embending

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.trans1 = nn.ConvTranspose2d(64, 64, kernel_size=3, padding=1)
        self.trans2 = nn.ConvTranspose2d(64, 32, kernel_size=3, padding=1, stride=2, output_padding=1)
        self.trans3 = nn.ConvTranspose2d(32, 32, kernel_size=3, padding=1)
        self.trans4= nn.ConvTranspose2d(32, 16, kernel_size=3, padding=1, stride=2, output_padding=1)
        self.trans5 = nn.ConvTranspose2d(16, 16, kernel_size=3, padding=1)
        self.trans6 = nn.ConvTranspose2d(16, 4, kernel_size=3, padding=1, stride=2, output_padding=1)

    def forward(self, latent_inputs, batch_size):
        x = torch.reshape(latent_inputs, (batch_size, 64, 8, 8))
        x = F.relu(self.trans1(x))
        x = F.relu(self.trans2(x))
        x = F.relu(self.trans3(x))
        x = F.relu(self.trans4(x))
        x = F.relu(self.trans5(x))
        decoder_outputs = F.relu(self.trans6(x))
        return decoder_outputs
         
class RandomCrop(object):
    """ Randomly crops raw and target image reespectively 
    Args: size(int) shape of new image (size,size) 
    """
    def __init__(self, size):
        self.size = size
    
    def __call__(self, raw, dslr):
        # print(raw.size())
        w, h = raw.shape[1:]
        i = np.random.randint(0, h - self.size)
        j = np.random.randint(0, w - self.size)
        cropped_raw = raw[:,i : i + self.size, j : j + self.size]
        cropped_dslr = dslr[:,i : i + self.size, j : j + self.size]
        return cropped_raw, cropped_dslr

import matplotlib.pyplot as plt

def read_target_image(path: str, size):
    image = cv2.imread(path)
    if image is None:
        raise Exception(f'Can not read image {path}')
    image = cv2.resize(image, size)
    image = image[:,:,::-1] #bgr -> rgb
    return image.astype(np.float32) / 255


def read_bayer_image(path: str):
    raw = np.asarray(imageio.imread(path))
    if raw is None:
        raise Exception(f'Can not read image {path}')
    ch_B  = raw[1::2, 1::2]
    ch_Gb = raw[0::2, 1::2]
    ch_R  = raw[0::2, 0::2]
    ch_Gr = raw[1::2, 0::2]
    combined = np.dstack((ch_B, ch_Gb, ch_R, ch_Gr))
    return combined.astype(np.float32) / (4 * 255)


def random_crop(image, size):
    h, w = image.shape[:2]
    x = np.random.randint(0, w - size[0])
    y = np.random.randint(0, h - size[1])
    return image[y:y+size[1], x:x+size[0]]

def plt_display(image, title):
    fig = plt.figure()
    a = fig.add_subplot(1, 1, 1)
    imgplot = plt.imshow(image)
    a.set_title(title)
IMG_SIZE = (64, 64)
BATCH_SIZE = 64
zurich_base_dir = "/alpha/gosha20777/Zurich-RAW-to-DSLR-Dataset"
mai_base_dir = "/alpha/gosha20777/MAI2021"
zurich_train_dataset = DatasetZurich(zurich_base_dir, 46839, istest=False, transform=RandomCrop(64))
zurich_test_dataset = DatasetZurich(zurich_base_dir, 1204, istest=True, transform=RandomCrop(64))
mai_dataser = DatasetMai(mai_base_dir, 24161, transform=RandomCrop(64))
mai_train_dataset, mai_test_dataset =  torch.utils.data.random_split(mai_dataser, 
                                                            [24161-1204, 1204])
train_zurich_loader = DataLoader(zurich_train_dataset, 
                                 BATCH_SIZE, 
                                 shuffle=True,
                                 drop_last=True,
                                 pin_memory=True,)
train_mai_loader = DataLoader(mai_train_dataset, 
                                 BATCH_SIZE, 
                                 shuffle=False,
                                 drop_last=True,
                                 pin_memory=True)

test_mai_loader = DataLoader(mai_test_dataset, 
                                 BATCH_SIZE, 
                                 shuffle=False,
                                 drop_last=True,
                                 pin_memory=True)
test_zurich_loader = DataLoader(zurich_test_dataset,
                                 BATCH_SIZE, 
                                 shuffle=False,
                                 drop_last=True,
                                 pin_memory=True)
from math import exp
def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()


def create_window(window_size, channel=1):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window


def ssim(img1, img2, window_size=11, window=None, size_average=True, full=False, val_range=None):
    # Value range can be different from 255. Other common ranges are 1 (sigmoid) and 2 (tanh).
    if val_range is None:
        if torch.max(img1) > 128:
            max_val = 255
        else:
            max_val = 1

        if torch.min(img1) < -0.5:
            min_val = -1
        else:
            min_val = 0
        L = max_val - min_val
    else:
        L = val_range

    padd = 0
    (_, channel, height, width) = img1.size()
    if window is None:
        real_size = min(window_size, height, width)
        window = create_window(real_size, channel=channel).to(img1.device)

    mu1 = F.conv2d(img1, window, padding=padd, groups=channel)
    mu2 = F.conv2d(img2, window, padding=padd, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=padd, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=padd, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=padd, groups=channel) - mu1_mu2

    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2

    v1 = 2.0 * sigma12 + C2
    v2 = sigma1_sq + sigma2_sq + C2
    cs = torch.mean(v1 / v2)  # contrast sensitivity

    ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)

    if size_average:
        ret = ssim_map.mean()
    else:
        ret = ssim_map.mean(1).mean(1).mean(1)

    if full:
        return ret, cs
    return ret


def msssim(img1, img2, window_size=11, size_average=True, val_range=None, normalize=False):
    device = img1.device
    weights = torch.FloatTensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333]).to(device)
    levels = weights.size()[0]
    mssim = []
    mcs = []
    for _ in range(levels):
        sim, cs = ssim(img1, img2, window_size=window_size, size_average=size_average, full=True, val_range=val_range)
        mssim.append(sim)
        mcs.append(cs)

        img1 = F.avg_pool2d(img1, (2, 2))
        img2 = F.avg_pool2d(img2, (2, 2))

    mssim = torch.stack(mssim)
    mcs = torch.stack(mcs)

    # Normalize (to avoid NaNs during training unstable models, not compliant with original definition)
    if normalize:
        mssim = (mssim + 1) / 2
        mcs = (mcs + 1) / 2

    pow1 = mcs ** weights
    pow2 = mssim ** weights
    # From Matlab implementation https://ece.uwaterloo.ca/~z70wang/research/iwssim/
    output = torch.prod(pow1[:-1] * pow2[-1])
    return output


# Classes to re-use window
class SSIM(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True, val_range=None):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.val_range = val_range

        # Assume 1 channel for SSIM
        self.channel = 1
        self.window = create_window(window_size)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.dtype == img1.dtype:
            window = self.window
        else:
            window = create_window(self.window_size, channel).to(img1.device).type(img1.dtype)
            self.window = window
            self.channel = channel

        return ssim(img1, img2, window=window, window_size=self.window_size, size_average=self.size_average)

class MSSSIM(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True, channel=3):
        super(MSSSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = channel

    def forward(self, img1, img2):
        # TODO: store window between calls if possible
        return msssim(img1, img2, window_size=self.window_size, size_average=self.size_average)
def mssim(y_true, y_pred):
    return 1.0 - ssim(y_pred, y_true)

mae_loss = torch.nn.L1Loss(reduction='mean')

def log_normal_pdf(sample, mean, logvar, raxis=1):
    log2pi = torch.log(torch.tensor(2. * np.pi))
    return torch.sum(
          -.5 * ((sample - mean) ** 2. * torch.exp(-logvar) + logvar + log2pi),
          dim=raxis)

def embedding_loss(y_pred):
    # print(y_pred.shape)
    mean, logvar = torch.split(y_pred, split_size_or_sections=128, dim=1)
    eps = torch.randn(BATCH_SIZE, mean.shape[1]).to(device=device)
    z = eps * torch.exp(logvar * .5) + mean
    logpz = log_normal_pdf(z, torch.zeros_like(mean), torch.zeros_like(logvar))
    logqz_x = log_normal_pdf(z, mean, logvar)
    return -torch.mean(logpz - logqz_x)
encoder_zurich = Encoder().to(device=device)
encoder_mai = Encoder().to(device=device)
decoder_zurich = Decoder().to(device=device)
decoder_mai = Decoder().to(device=device)


# params = list(encoder_mai.parameters()) + list(encoder_zurich.parameters()) + list(decoder_mai.parameters()) + list(decoder_zurich.parameters())
# optimizer = torch.optim.Adam(params(), lr=10**-3, betas=(0.9, 0.999))
from itertools import chain
params = chain(encoder_zurich.parameters(), encoder_mai.parameters(),
      decoder_mai.parameters(), decoder_zurich.parameters())
optimizer = torch.optim.Adam(params, lr=10**-3, betas=(0.9, 0.999))
wandb.init(
        project="2 unet")
from utils import to_psnr, to_ssim_skimage
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,patience=5)
from itertools import cycle

for epoch in range(20):
    # psnr_list = []

    encoder_mai.train()
    encoder_zurich.train()
    decoder_mai.train()
    decoder_zurich.train()

    for data_zurich, data_mai in zip(train_zurich_loader, cycle(train_mai_loader)):
        x_zurich, target_zurich, _ = data_zurich
        x_mai, target_mai, _ = data_mai
        x_mai = x_mai.to(device=device)
        x_zurich = x_zurich.to(device=device)
        target_mai = target_mai.to(device=device)
        target_zurich = target_zurich.to(device=device)
        # print(x_zurich.shape, x_mai.shape)
        latent_layer_zurich, embedding_zurich = encoder_zurich(x_zurich)
        latent_layer_mai, embedding_mai =  encoder_mai(x_mai)
        out_zurich = decoder_zurich(latent_layer_zurich, BATCH_SIZE)
        out_mai = decoder_mai(latent_layer_mai, BATCH_SIZE)

        optimizer.zero_grad()
        loss = embedding_loss(embedding_mai) + embedding_loss(embedding_zurich)
        loss = loss + mssim(out_mai, x_mai) + mssim(out_zurich, x_zurich)
        loss += mae_loss(embedding_mai, embedding_zurich)
        loss.backward()
        optimizer.step()
        print(to_ssim_skimage(out_zurich, x_zurich)[0], loss)
        metrics = {
            'train/loss': loss,
            # 'train/ssim_mai': torch.tensor(to_ssim_skimage(out_mai, x_mai)[0]),
            # 'train/ssim_zuroich': to_ssim_skimage(out_zurich, x_zurich)[0],
            # 'train/psnr_mai': to_psnr(out_mai, x_mai)[0],
            # 'train/psnr_zurich': to_psnr(out_zurich, x_zurich)[0],
            'train/epoch': epoch
        }
        wandb.log(metrics)
        scheduler.step(loss)

    encoder_mai.eval()
    encoder_zurich.eval()
    decoder_mai.eval()
    decoder_zurich.eval()

    i = 0

    for data_zurich, data_mai in zip(test_zurich_loader, cycle(test_mai_loader)):
        with torch.no_grad():
            x_zurich, target_zurich, _ = data_zurich
            x_mai, target_mai, _ = data_mai
            x_mai = x_mai.to(device=device)
            x_zurich = x_zurich.to(device=device)
            target_mai = target_mai.to(device=device)
            target_zurich = target_zurich.to(device=device)
            
            latent_layer_zurich, embedding_zurich = encoder_zurich(x_zurich)
            latent_layer_mai, embedding_mai =  encoder_mai(x_mai)
            out_zurich = decoder_zurich(latent_layer_zurich, BATCH_SIZE)
            out_mai = decoder_mai(latent_layer_mai, BATCH_SIZE)

            loss = 0.2*embedding_loss(embedding_mai) + 0.2*embedding_loss(embedding_zurich)
            loss = loss + 0.2*mssim(out_mai, x_mai) + 0.2*mssim(out_zurich, x_zurich)
            loss += 0.2*mae_loss(embedding_mai, embedding_zurich)
            i += 1
            metrics = {
                'val/loss': loss,
                'val/i': i,
                'val/ssim_mai': to_ssim_skimage(out_mai, x_mai),
                'val/ssim_zuroich': to_ssim_skimage(out_zurich, x_zurich),
                'val/psnr_mai': to_psnr(out_mai, x_mai),
                'val/psnr_zurich': to_psnr(out_zurich, x_zurich),
                'val/epoch': epoch
            }
            wandb.log(metrics)        

        
wandb.stop()
torch.save(encoder_mai.state_dict(), './mai_encoder.pkl')
torch.save(encoder_zurich.state_dict(),'./zurich_encoder.pkl')
torch.save(decoder_mai.state_dict(), './mai_decoder.pkl')
torch.save(decoder_zurich.state_dict(),'./zurich_decoder.pkl')