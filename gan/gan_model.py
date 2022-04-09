import torch
import torch.nn as nn
import torch.nn.functional as F

import py_func.models_func as my_model

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