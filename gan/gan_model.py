import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg19

import py_func.models_func as my_model

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

class Discriminator(nn.Module):
    """
    Discriminatorのクラス
    """
    def __init__(self, input_shape):
        super(Discriminator, self).__init__()
                
        self.input_shape = input_shape
        in_channels, in_height, in_width = self.input_shape
        patch_h, patch_w = int(in_height / 2 ** 4), int(in_width / 2 ** 4)
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

