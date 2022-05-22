import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg19

import py_func.models_func as my_model

##########---------pix2pix---------##########

#submodule
class LeakyReLU_Conv(nn.Module):
    def __init__(self, inc, outc, ks, strd, pad):
        self.CNN = nn.Sequential(
            nn.LeakyReLU(0.2),
            nn.Conv2d(inc, outc, ks, strd, pad),
        )
    def forward(self, x):
        x = self.CNN(x)
        return x

class Unet_encoder(nn.Module):
    def __init__(self, inc, outc, norm_true = True):
        super().__init__()
        layer = [LeakyReLU_Conv(inc, outc, 4, 2, 1)]
        if norm_true:
            layer.append(nn.BatchNorm2d(outc))
        self.model = nn.Sequential(*layer)
    def forward(self, x):
        x = self.model(x)
        return x

class ReLU_Convtp(nn.Module):
    def __init__(self, inc, outc, ks, strd, pad):
        super().__init__()
        self.CNN = nn.Sequential(
            nn.ReLU(),
            nn.ConvTranspose2d(inc, outc, ks, strd, pad)
        )
    def forward(self, x):
        x = self.CNN(x)
        return x

class Unet_decoder(nn.Module):
    def __init__(self, inc, outc, norm_true = True, drop_true = False):
        super().__init__()
        layer = [ReLU_Convtp(inc, outc, 4, 2, 1)]
        if norm_true:
            layer.append(nn.BatchNorm2d(outc))
        if drop_true:
            layer.append(nn.Dropout(0.5))
        self.model = nn.Sequential(*layer)
    def forward(self, x):
        x = self.model(x)
        return x

#model
class Generator_Unet(nn.Module):
    def __init__(self):
        self.e0 = nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1)
        self.e1 = Unet_encoder(64, 128)
        self.e2 = Unet_encoder(128, 256)
        self.e3 = Unet_encoder(256, 512)
        self.e4 = Unet_encoder(512, 512)
        self.e5 = Unet_encoder(512, 512)
        self.e6 = Unet_encoder(512, 512)
        self.e7 = Unet_encoder(512, 512, False)

        self.d7 = Unet_decoder(512, 512)
        self.d6 = Unet_decoder(1024, 512, drop_true = True)
        self.d5 = Unet_decoder(1024, 512, drop_true = True)
        self.d4 = Unet_decoder(1024, 512, drop_true = True)
        self.d3 = Unet_decoder(1024, 256)
        self.d2 = Unet_decoder(512, 128)
        self.d1 = Unet_decoder(256, 64)

        self.d0 = nn.Sequential(
            Unet_decoder(128, 3, norm_true=False),
            nn.Tanh(),
        )
    def concat(self, x, y):
        # 特徴マップの結合
        return torch.cat([x, y], dim=1)

    def forward(self, x):
        # 偽物画像の生成
        x0 = self.e0(x)
        x1 = self.e1(x0)
        x2 = self.e2(x1)
        x3 = self.e3(x2)
        x4 = self.e4(x3)
        x5 = self.e5(x4)
        x6 = self.e6(x5)
        x7 = self.e7(x6)
        y7 = self.d7(x7)
        # Encoderの出力をDecoderの入力にSkipConnectionで接続
        y6 = self.d6(self.concat(x6, y7))
        y5 = self.d5(self.concat(x5, y6))
        y4 = self.d4(self.concat(x4, y5))
        y3 = self.d3(self.concat(x3, y4))
        y2 = self.d2(self.concat(x2, y3))
        y1 = self.d1(self.concat(x1, y2))
        y0 = self.d0(self.concat(x0, y1))

        return y0
    
class Discriminatorpix2pix(nn.Module):
    def __init__(self):
        super().__init__()
        self.disc = nn.Sequential(
            my_model.Conv_LeakyReLU(6, 64, 4, 2, 1, 0.2),
            my_model.Conv_Bn_LeakyReLU(64, 128, 4, 2, 1, 0.2),
            my_model.Conv_Bn_LeakyReLU(128, 256, 4, 2, 1, 0.2),
            my_model.Conv_Bn_LeakyReLU(256, 512, 4, 1, 1, 0.2),
            nn.Conv2d(512, 1, 4, 1, 1)
        )
    def forward(self, x):
        x = self.disc(x)
        return x



###########---------ESRGAN---------##########
#submodule
class DenseResidualBlock(nn.Module):
    """
    :param length: number of Convlayer
    :type length: int
    :param filters: fundamental number of channels
    :type filters: int
    :param ngsllist: list consist of negative slope of each layers's LeakyReLU 
    :type ngsllist: list
    :param res_scale: coefficient of output
    :type res_scale: float or double
    """
    def __init__(self, length, filters, res_scale=0.2):
        super().__init__()
        self.lenlayer = length
        self.res_scale = res_scale


        convs = []
        for i in range(self.lenlayer):            
            if(i < self.lenlayer-1):
                #最終層以外はConv_LeakyReLU
                #データサイズを変えないためにカーネルサイズとストライド、パディングは固定
                convs.append(my_model.Conv_LeakyReLU(filters*(i+1), filters, \
                3, 1, 1, 0.2))
            else:
                #最終層はConvのみ
                convs.append(nn.Conv2d(filters*(i+1), filters, 3, 1, 1))
        
        self.convs = nn.Sequential(*convs)

    def forward(self, x):
        inputs = x
        for convlay in self.convs:
            out = convlay(inputs)
            inputs = torch.cat([inputs, out], 1)
        return out.mul(self.res_scale) + x

class DiscriminatorBlock(nn.Module):
    def __init__(self, inc, outc, first_block = False):
        super(DiscriminatorBlock, self).__init__()

        layers = []

        if not first_block:
            layers.append(my_model.Conv_Bn_LeakyReLU(inc, outc, 3, 1, 1, 0.2))
        else:
            layers.append(my_model.Conv_LeakyReLU(inc, outc, 3, 1, 1, 0.2))

        layers.append(my_model.Conv_Bn_LeakyReLU(outc, outc, 3, 2, 1, 0.2))
        self.discblock = nn.Sequential(*layers)
    def forward(self, x):
        x = self.discblock(x)
        return x

class ResidualInResidualDenseBlock(nn.Module):
    """
    GenearatorのResidualInResidualDenseBlockのクラス
    """
    def __init__(self, lendns, fltrsdns,\
         length = 3, res_scale=0.2):
        super(ResidualInResidualDenseBlock, self).__init__()
        self.res_scale = res_scale

        dnsblck = []
        for _ in range(length):
            dnsblck.append(DenseResidualBlock(lendns, fltrsdns))
        self.dnsblck = nn.Sequential(*dnsblck)
    
    def forward(self, x):
        return self.dnsblck(x).mul(self.res_scale) + x

#Generator
class GeneratorRRDB(nn.Module):
    """
    Generatorのクラス
    """
    def __init__(self, inc, fltrs, lendns, 
    num_res_blck=16, num_upsmpl=2, upscale_factor=2):
        super(GeneratorRRDB, self).__init__()
        
        self.conv1 = nn.Conv2d(inc, fltrs, 3, 1, 1)
        
        self.res_blocks = nn.Sequential(*[ResidualInResidualDenseBlock(
            lendns, fltrs
            ) for _ in range(num_res_blck)])
        
        self.conv2 = nn.Conv2d(fltrs, fltrs, 3, 1, 1)
        
        upsample_layers = []
        for _ in range(num_upsmpl):
            upsample_layers += [
                nn.Conv2d(fltrs, fltrs*(upscale_factor**2), 3, 1, 1),
                nn.LeakyReLU(),
                nn.PixelShuffle(upscale_factor),
            ]
        self.upsampling = nn.Sequential(*upsample_layers)

        self.conv3 = nn.Sequential(
            nn.Conv2d(fltrs, fltrs, 3, 1, 1),
            nn.LeakyReLU(),
            nn.Conv2d(fltrs, inc, 3, 1, 1),
        )
    
    def forward(self, x):
        out1 = self.conv1(x)
        out = self.res_blocks(out1)
        out2 = self.conv2(out)
        #各要素足し算
        out = torch.add(out1, out2)
        out = self.upsampling(out)
        out = self.conv3(out)
        return out

#特徴量抽出器
class FeatureExtractor(nn.Module):
    """
    Perceputual lossを計算するために特徴量を抽出するためのクラス
    """
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        vgg19_model = vgg19(pretrained=True)
        self.vgg19_54 = nn.Sequential(*list(
            vgg19_model.features.children())[:35])

    def forward(self, img):
        return self.vgg19_54(img)

#ESRの識別器
class DiscriminatorESR(nn.Module):
    """
    Discriminatorのクラス
    """
    def __init__(self, input_shape):
        super(Discriminator, self).__init__()
                
        self.input_shape = input_shape
        in_channels, in_height, in_width = self.input_shape
        patch_h, patch_w = int(in_height / 2 ** 4), int(in_width / 2 ** 4)
        #学習時に識別器の正解ラベルを作るために必要
        self.output_shape = (1, patch_h, patch_w)
    
        layers = []
        in_filters = in_channels
        for i, out_filters in enumerate([64, 128, 256, 512]):
            layers.append(DiscriminatorBlock(in_filters, out_filters, first_block=(i==0)))
            in_filters = out_filters
        
        layers.append(nn.Conv2d(out_filters, 
                                1, 
                                kernel_size=3, 
                                stride=1, 
                                padding=1))
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, img):
        return self.model(img)






#######-----------GAN,CGAN,LSGAN------------#########

class Generator(nn.Module):
    """
    生成器Gのクラス
    """
    def __init__(self, chlist, kslist, strdlist, padlist, opadlist):
        """
        :param chlist: チャネル数

        length of chlist is length of kslist + 1.
        first element of chlist is dimension of noize.

        """
        super(Generator, self).__init__()

        # ニューラルネットワークの構造を定義する
        Deconv_list = []


        #最終層以外はConvtp_Bn_ReLU
        for i in range(len(kslist) - 1):
            Deconv_list.append(my_model.Convtp_Bn_ReLu(chlist[i], chlist[i + 1], 
                                                      kslist[i], strdlist[i],
                                                      padlist[i], opadlist[i]
                                                      )
                               )
        
        #最終層はConvtp_Tanh
        last = len(kslist) - 1
        Deconv_list.append(my_model.Convtp_Tanh(chlist[last], chlist[last + 1], 
                                              kslist[last], strdlist[last],
                                              padlist[last], opadlist[last]
                                              )
                          )
        self.Deconvs = nn.Sequential(*Deconv_list)
        

    def forward(self, z):
        """
        順方向の演算
        :param z: 入力ベクトル
        :return: 生成画像
        """
        z = self.Deconvs(z)
        return z

class Discriminator(nn.Module):
    """
    識別器Dのクラス
    """
    def __init__(self, chlist, kslist, strdlist, padlist, ngsllist):
        """
        :param chlist: チャネル数

        length of chlist is length of kslist + 1.
        """
        super(Discriminator, self).__init__()

        # ニューラルネットワークの構造を定義する
        conv_list = []

        #入力層はBnなし
        conv_list.append(my_model.Conv_LeakyReLU(chlist[0], chlist[1], kslist[0], strdlist[0], padlist[0], ngsllist[0]))

        #入力層と最終層以外はLeakyReLU
        for i in range(1, len(kslist) - 1):
            conv_list.append(my_model.Conv_Bn_LeakyReLU(chlist[i], chlist[i + 1], 
                                                      kslist[i], strdlist[i],
                                                      padlist[i], ngsllist[i]
                                                      )
                            )

        #最終層はSigmoid
        last = len(kslist) - 1
        conv_list.append(my_model.Conv_Sigmoid(chlist[last], chlist[last + 1], 
                                              kslist[last], strdlist[last],
                                              padlist[last]
                                              )
                        )
        self.Convs = nn.Sequential(*conv_list)


    def forward(self, x):
        """
        順方向の演算
        :param x: 本物画像あるいは生成画像
        :return: 識別信号
        """
        x = self.Convs(x)
        return x.squeeze()     # Tensorの形状を(B)に変更して戻り値とする    

class Dscrmntr_notsigmoid(nn.Module):
    """
    識別器Dのクラス
    """
    def __init__(self, chlist, kslist, strdlist, padlist, ngsllist):
        """
        :param chlist: チャネル数

        length of chlist is length of kslist + 1.
        """
        super(Dscrmntr_notsigmoid, self).__init__()

        # ニューラルネットワークの構造を定義する
        conv_list = []

        #入力層はBnなし
        conv_list.append(my_model.Conv_LeakyReLU(chlist[0], chlist[1], kslist[0], strdlist[0], padlist[0], ngsllist[0]))

        #入力層と最終層以外はLeakyReLU
        for i in range(1, len(kslist) - 1):
            conv_list.append(my_model.Conv_Bn_LeakyReLU(chlist[i], chlist[i + 1], 
                                                      kslist[i], strdlist[i],
                                                      padlist[i], ngsllist[i]
                                                      )
                            )

        #最終層はSigmoid
        last = len(kslist) - 1
        conv_list.append(nn.Conv2d(chlist[last], chlist[last + 1], 
                                    kslist[last], strdlist[last],
                                    padlist[last]
                                    )
                        )
        self.Convs = nn.Sequential(*conv_list)


    def forward(self, x):
        """
        順方向の演算
        :param x: 本物画像あるいは生成画像
        :return: 識別信号
        """
        x = self.Convs(x)
        return x.squeeze()     # Tensorの形状を(B)に変更して戻り値とする
