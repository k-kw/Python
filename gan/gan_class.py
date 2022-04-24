import torch
from torchvision.utils import save_image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import json


import py_func.gan.gan_model as ganmd
import numpy as np
from PIL import Image
from glob import glob
import os
import sys
import os.path as osp

#-------------------ESRGAN-------------------
class ImageDataset(Dataset):
    """
    学習のためのDatasetクラス
    32x32の低解像度の本物画像と、
    128x128の本物画像を出力する
    """
    def __init__(self, dataset_dir, hr_shape, mean, std):
        hr_height, hr_width = hr_shape
        
        # 低解像度の画像を取得するための処理
        self.lr_transform = transforms.Compose([
            transforms.Resize((hr_height // 4, hr_height // 4), Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)])

        # 高像度の画像を取得するための処理
        self.hr_transform = transforms.Compose([
            transforms.Resize((hr_height, hr_height), Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)])
        
        self.files = sorted(glob(osp.join(dataset_dir, '*')))
    
    def __getitem__(self, index):
        img = Image.open(self.files[index % len(self.files)])
        img_lr = self.lr_transform(img)
        img_hr = self.hr_transform(img)
        
        return {'lr': img_lr, 'hr': img_hr}
    
    def __len__(self):
        return len(self.files)

class TestImageDataset(Dataset):
    """
    Generatorによる途中経過の確認のためのDatasetクラス
    lr_transformで入力画像を高さと幅それぞれ1/4の低解像度の画像を生成し、
    hr_transformでオリジナルの画像を高解像度の画像として用いる
    """
    def __init__(self, dataset_dir, mean, std):
        self.hr_transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean, std)])
        self.files = sorted(glob(osp.join(dataset_dir, '*')))
    
    def lr_transform(self, img, img_size, mean, std):
        """
        様々な入力画像のサイズに対応するために、
        入力画像のサイズを1/4にするように処理
        """
        img_width, img_height = img_size
        self.__lr_transform = transforms.Compose([
            transforms.Resize((img_height // 4, 
                               img_width // 4), 
                               Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)])
        img = self.__lr_transform(img)
        return img
            
    def __getitem__(self, index):
        img = Image.open(self.files[index % len(self.files)])
        img_size = img.size
        img_lr = self.lr_transform(img, img_size)
        img_hr = self.hr_transform(img)        
        return {'lr': img_lr, 'hr': img_hr}
    
    def __len__(self):
        return len(self.files)

def denormalize(tensors, std, mean):
  """
  高解像度の生成画像の非正規化を行う
  """
  for c in range(3):
    tensors[:, c].mul_(std[c]).add_(mean[c])
  return torch.clamp(tensors, 0, 255)

class ESRGAN():
    """
    ESRGANの処理を実装するクラス
    optに様々なパラメータ
    """
    def __init__(self, opt, log_dir):
        self.generator = ganmd.GeneratorRRDB(opt.channels, fltrs=64, lendns = 5, \
            num_res_blck=opt.residual_blocks, num_upsmpl=2, upscale_factor=2).to(opt.device)
        
        self.discriminator = ganmd.Discriminator(input_shape=(opt.channels, opt.hr_height, opt.hr_width)).to(opt.device)

        self.feature_extractor = ganmd.FeatureExtractor().to(opt.device)
        self.feature_extractor.eval()

        self.criterion_GAN = nn.BCEWithLogitsLoss().to(opt.device)
        self.criterion_content = nn.L1Loss().to(opt.device)
        self.criterion_pixel = nn.L1Loss().to(opt.device)

        self.optimizer_G = optim.Adam(self.generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
        self.optimizer_D = optim.Adam(self.discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

        self.Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor
        self.writer = SummaryWriter(log_dir=log_dir)
        self.opt = opt
    
    def pre_train(self, imgs, batches_done, batch_num, epoch):
        """
        loss pixelのみで事前学習を行う
        """
        # preprocess
        imgs_lr = Variable(imgs['lr'].type(self.Tensor))
        imgs_hr = Variable(imgs['hr'].type(self.Tensor))

        # ground truth
        valid = Variable(self.Tensor(np.ones((imgs_lr.size(0), *self.discriminator.output_shape))), 
                          requires_grad=False)
        fake = Variable(self.Tensor(np.zeros((imgs_lr.size(0), *self.discriminator.output_shape))), 
                        requires_grad=False)

        # バックプロパゲーションの前に勾配を0にする
        self.optimizer_G.zero_grad()

        # 低解像度の画像から高解像度の画像を生成
        gen_hr = self.generator(imgs_lr)

        loss_pixel = self.criterion_pixel(gen_hr, imgs_hr)

        # 画素単位の損失であるloss_pixelで事前学習を行う
        loss_pixel.backward()
        self.optimizer_G.step()
        train_info = {'epoch': epoch, 'batch_num': batch_num, 'loss_pixel': loss_pixel.item()}
        if batch_num == 1:
            sys.stdout.write('\n{}'.format(train_info))
        else:
            sys.stdout.write('\r{}'.format('\t'*20))
            sys.stdout.write('\r{}'.format(train_info))
        sys.stdout.flush()

        self.save_loss(train_info, batches_done)

    def train(self, imgs, batches_done, batch_num, epoch):
        """
        pixel loss以外の損失も含めて本学習を行う
        """
        # 前処理
        imgs_lr = Variable(imgs['lr'].type(self.Tensor))
        imgs_hr = Variable(imgs['hr'].type(self.Tensor))

        # 正解ラベル
        valid = Variable(self.Tensor(np.ones((imgs_lr.size(0), *self.discriminator.output_shape))), 
                          requires_grad=False)
        fake = Variable(self.Tensor(np.zeros((imgs_lr.size(0), *self.discriminator.output_shape))), 
                        requires_grad=False)

        # 低解像度の画像から高解像度の画像を生成
        self.optimizer_G.zero_grad()
        gen_hr = self.generator(imgs_lr)

        # Pixel loss
        loss_pixel = self.criterion_pixel(gen_hr, imgs_hr)

        # 推論
        pred_real = self.discriminator(imgs_hr).detach()
        pred_fake = self.discriminator(gen_hr)

        # Adversarial loss
        loss_GAN = self.criterion_GAN(pred_fake - pred_real.mean(0, keepdim=True), valid)

        # Perceptual loss
        gen_feature = self.feature_extractor(gen_hr)
        real_feature = self.feature_extractor(imgs_hr).detach()
        loss_content = self.criterion_content(gen_feature, real_feature)

        # 生成器のloss
        loss_G = loss_content + self.opt.lambda_adv * loss_GAN + self.opt.lambda_pixel * loss_pixel
        loss_G.backward()
        self.optimizer_G.step()

        # 識別機のLoss
        self.optimizer_D.zero_grad()
        pred_real = self.discriminator(imgs_hr)
        pred_fake = self.discriminator(gen_hr.detach())

        loss_real = self.criterion_GAN(pred_real - pred_fake.mean(0, keepdim=True), valid)            
        loss_fake = self.criterion_GAN(pred_fake - pred_real.mean(0, keepdim=True), fake)    
        loss_D = (loss_real + loss_fake) / 2
        loss_D.backward()
        self.optimizer_D.step()

        train_info = {'epoch': epoch, 'batch_num': batch_num,  'loss_D': loss_D.item(), 'loss_G': loss_G.item(),
                      'loss_content': loss_content.item(), 'loss_GAN': loss_GAN.item(), 'loss_pixel': loss_pixel.item(),}
        if batch_num == 1:
            sys.stdout.write('\n{}'.format(train_info))
        else:
            sys.stdout.write('\r{}'.format('\t'*20))
            sys.stdout.write('\r{}'.format(train_info))
        sys.stdout.flush()

        self.save_loss(train_info, batches_done)

    def save_loss(self, train_info, batches_done):
        """
        lossの保存
        """
        for k, v in train_info.items():
            self.writer.add_scalar(k, v, batches_done)

    def save_image(self, imgs, batches_done, image_test_save_dir, idx):
        """
        画像の保存
        """
        with torch.no_grad():
            # Save image grid with upsampled inputs and outputs
            imgs_lr = Variable(imgs["lr"].type(self.Tensor))
            gen_hr = self.generator(imgs_lr)
            gen_hr = denormalize(gen_hr)
            self.writer.add_image('image_{}'.format(idx), gen_hr[0], batches_done)

            image_batch_save_dir = osp.join(image_test_save_dir, '{:03}'.format(idx))
            # gen_hr_dir = osp.join(image_batch_save_dir, "hr_image")
            os.makedirs(image_batch_save_dir, exist_ok=True)
            save_image(gen_hr, osp.join(image_batch_save_dir, "{:09}.png".format(batches_done)), nrow=1, normalize=False)

    def save_weight(self, batches_done, weight_save_dir):
        """
        重みの保存
        """
        # Save model checkpoints
        generator_weight_path = osp.join(weight_save_dir, "generator_{:08}.pth".format(batches_done))
        discriminator_weight_path = osp.join(weight_save_dir, "discriminator_{:08}.pth".format(batches_done))

        torch.save(self.generator.state_dict(), generator_weight_path)
        torch.save(self.discriminator.state_dict(), discriminator_weight_path)


def save_json(file, save_path, mode):
    """Jsonファイルを保存
    """
    with open(save_path, mode) as outfile:
        json.dump(file, outfile, indent=4)