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

#-------------------pix2pix------------------
import random

# 条件画像と正解画像のペアデータセット生成クラス
class AlignedDataset(Dataset):
    IMG_EXTENSIONS = ['.png', 'jpg']
    # configは全ての学習条件を格納する

    # 画像データは'/path/to/data/train'および'/path/to/data/test'に
    # {A,B}の形式で格納されているものとみなす

    def __init__(self, config):
        # データセットクラスの初期化
        self.config = config

        # データディレクトリの取得
        dir = os.path.join(config.dataroot, config.phase)
        # 画像データパスの取得
        self.AB_paths = sorted(self.__make_dataset(dir))

    #インスタンス化していなくても呼び出せるメソッドには@classmethodをつける
    @classmethod
    def is_image_file(self, fname):
        # 画像ファイルかどうかを返す
        return any(fname.endswith(ext) for ext in self.IMG_EXTENSIONS)

    @classmethod
    def __make_dataset(self, dir):
        # 画像データセットをメモリに格納
        images = []
        assert os.path.isdir(dir), '%s is not a valid directory' % dir

        # os.walkでディレクトリを上位階層から順にファイルをfnamesに
        for root, _, fnames in sorted(os.walk(dir)):
            for fname in fnames:
                if self.is_image_file(fname):
                    path = os.path.join(root, fname)
                    images.append(path)
        return images

    def __transform(self, param):
        list = []

        load_size = self.config.load_size

        # 入力画像を一度286x286にリサイズし、その後で256x256にランダムcropする
        list.append(transforms.Resize([load_size, load_size], Image.BICUBIC))

        (x, y) = param['crop_pos']
        crop_size = self.config.crop_size
        list.append(transforms.Lambda(lambda img: img.crop((x, y, x + crop_size, y + crop_size))))

        # 1/2の確率で左右反転する
        if param['flip']:
            list.append(transforms.Lambda(lambda img: img.transpose(Image.FLIP_LEFT_RIGHT)))

        # RGB画像をmean=(0.5,0.5,0.5), std=(0.5,0.5,0.5)にNormalizeする
        list += [transforms.ToTensor(),
                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]

        return transforms.Compose(list)

    def __transform_param(self):
        x_max = self.config.load_size - self.config.crop_size
        x = random.randint(0, np.maximum(0, x_max))
        y = random.randint(0, np.maximum(0, x_max))

        flip = random.random() > 0.5

        return {'crop_pos': (x, y), 'flip': flip}

    def __getitem__(self, index):
        # 学習用データ１つの生成
        # A(テンソル) : 条件画像
        # B(テンソル) : Aのペアとなるターゲット画像

        # ランダムなindexの画像を取得 
        AB_path = self.AB_paths[index]
        AB = Image.open(AB_path).convert('RGB')

        # 画像を2分割してAとBをそれぞれ取得
        # ランダムシードの生成
        param = self.__transform_param()
        w, h = AB.size
        w2 = int(w / 2)
        # 256x256サイズの画像生成
        # 一度リサイズしてランダムな位置で256x256にcropする
        # AとBは同じ位置からcropする
        transform = self.__transform(param)
        A = transform(AB.crop((0, 0, w2, h)))
        B = transform(AB.crop((w2, 0, w, h)))

        #return {'A': A, 'B': B, 'A_paths': AB_path, 'B_paths': AB_path}
        return {'A': B, 'B': A, 'A_paths': AB_path, 'B_paths': AB_path}

    def __len__(self):
        # 全画像ファイル数を返す
        return len(self.AB_paths)

# GANのAdversarial損失の定義(Real/Fake識別)
class GANLoss(nn.Module):
    def __init__(self):
        super().__init__()

        self.register_buffer('real_label', torch.tensor(1.0))
        self.register_buffer('fake_label', torch.tensor(0.0))
        # Real/Fake識別の損失を、シグモイド＋バイナリクロスエントロピーで計算
        self.loss = nn.BCEWithLogitsLoss()

    def __call__(self, prediction, is_real):
        if is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label

        return self.loss(prediction, target_tensor.expand_as(prediction))





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
        self.mean = mean
        self.std = std
    
    def lr_transform(self, img, img_size):
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
            transforms.Normalize(self.mean, self.std)])
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

        #モデルを準備
        self.generator = ganmd.GeneratorRRDB(opt.channels, fltrs=64, lendns = 5, \
            num_res_blck=opt.residual_blocks, num_upsmpl=2, upscale_factor=2).to(opt.device)
        
        self.discriminator = ganmd.DiscriminatorESR(input_shape=(opt.channels, opt.hr_height, opt.hr_width)).to(opt.device)

        self.feature_extractor = ganmd.FeatureExtractor().to(opt.device)
        #特徴抽出器は学習しない
        self.feature_extractor.eval()

        #損失関数を準備
        self.criterion_GAN = nn.BCEWithLogitsLoss().to(opt.device)
        self.criterion_content = nn.L1Loss().to(opt.device)
        self.criterion_pixel = nn.L1Loss().to(opt.device)

        #最適化アルゴリズムを準備
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

        # # ground truth
        # valid = Variable(self.Tensor(np.ones((imgs_lr.size(0), *self.discriminator.output_shape))), 
        #                   requires_grad=False)
        # fake = Variable(self.Tensor(np.zeros((imgs_lr.size(0), *self.discriminator.output_shape))), 
        #                 requires_grad=False)

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
        # Relativistic Average Discriminator
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

        # Relativistic Average Discriminator
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

    def save_image(self, imgs, batches_done, image_test_save_dir, idx, mean, std):
        """
        画像の保存
        """
        with torch.no_grad():
            # Save image grid with upsampled inputs and outputs
            imgs_lr = Variable(imgs["lr"].type(self.Tensor))
            gen_hr = self.generator(imgs_lr)
            gen_hr = denormalize(gen_hr, mean, std)
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