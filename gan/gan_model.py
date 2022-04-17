import torch
import torch.nn as nn
import torch.nn.functional as F

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
    def __init__(self, length, filters, ngsllist, res_scale=0.2):
        super().__init__()
        self.lenlayer = length
        self.res_scale = res_scale


        convs = []
        for i in range(self.lenlayer):            
            if(i < self.lenlayer-1):
                #最終層以外はConv_LeakyReLU
                #データサイズを変えないためにカーネルサイズとストライド、パディングは固定
                convs.append(my_model.Conv_LeakyReLU(filters*(i+1), filters, \
                3, 1, 1, ngsllist[i]))
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
        


class ResidualInResidualDenseBlock(nn.Module):
    """
    GenearatorのResidualInResidualDenseBlockのクラス
    """
    def __init__(self, lendns, fltrsdns, ngsldns, rscldns,\
         length, res_scale=0.2):
        super(ResidualInResidualDenseBlock, self).__init__()
        self.res_scale = res_scale

        dnsblck = []
        for _ in range(length):
            dnsblck.append(DenseResidualBlock(lendns, fltrsdns, ngsldns, rscldns))
        self.dnsblck = nn.Sequential(*dnsblck)
    
    def forward(self, x):
        return self.dnsblck(x).mul(self.res_scale) + x