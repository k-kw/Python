import torch
import torch.nn as nn
import torch.nn.functional as F
torch.manual_seed(0)


#重み初期化関数, model.apply(def)で使う




#sim
#decode_model

#submodule

class Conv_ReLU_Pixshuf(nn.Module):
    def __init__(self, inc, outc, ks, strd, pad, upsclfac):
        super().__init__()
        self.pixshuf = nn.Sequential(
            nn.Conv2d(inc, outc, ks, strd, pad),
            nn.ReLU(),
            nn.PixelShuffle(upscale_factor = upsclfac),

        )
    
    def forward(self, x):
        x = self.pixshuf(x)
        return x

class Convtp_Tanh(nn.Module):
    def __init__(self, inc, outc, ks, strd, pad, outpad):
        super().__init__()
        self.Deconv = nn.Sequential(nn.ConvTranspose2d(in_channels = inc, out_channels = outc, 
                                                   kernel_size = ks, stride = strd, padding = pad, output_padding = outpad),
                                                   nn.Tanh(),
                                                   )
    
    def forward(self, x):
        x = self.Deconv(x)
        return x


class Convtp_Bn_ReLu(nn.Module):
  def __init__(self,input_channel,output_channel,kernel_size,stride,padding,output_padding):
    super().__init__()
    self.Deconv = nn.Sequential(nn.ConvTranspose2d(in_channels=input_channel, out_channels=output_channel, 
                                                   kernel_size=kernel_size,stride=stride,padding=padding,output_padding=output_padding),
                             nn.BatchNorm2d(output_channel),
                             nn.ReLU(),
                             )
  def forward(self,x):
    x = self.Deconv(x)
    return x


class Convtp_Bn_Sigmoid(nn.Module):
    def __init__(self, ic, oc, ks, strd, pad, outpad):
        super().__init__()
        self.Deconv = nn.Sequential(nn.ConvTranspose2d(in_channels = ic, out_channels = oc, kernel_size = ks, stride = strd, padding = pad, output_padding = outpad),
                                    nn.BatchNorm2d(oc), 
                                    nn.Sigmoid(),
                                    )
    def forward(self,x):
        x = self.Deconv(x)
        return x



#model



class decoder_outlayersigmoid(nn.Module):
    def __init__(self, input_size, fc1out, firstw, firsth, kslist, strdlist, padlist, opadlist, iclist, oclist):
        super().__init__()
        self.fc1 = nn.Linear(input_size,fc1out)
        self.relu = nn.ReLU()

        self.Deconv_list = []
        #最終層以外はConvtp_Bn_ReLu
        for i in range(len(kslist) - 1):
            self.Deconv_list.append(Convtp_Bn_ReLu(iclist[i],oclist[i],kslist[i],strdlist[i],
                                             padlist[i],opadlist[i]))
        #最終層はConvtp_Bn_Sigmoid
        last = len(kslist) - 1
        self.Deconv_list.append(Convtp_Bn_Sigmoid(iclist[last], oclist[last], kslist[last], strdlist[last],
                                             padlist[last], opadlist[last]))


        self.Deconvs = nn.Sequential(*self.Deconv_list)

        self.first_width = firstw
        self.first_height = firsth
    

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = x.view(-1, 1, self.first_height, self.first_width)
        x = self.Deconvs(x)
        return x


class simnet_decoder_allsize(nn.Module):
  def __init__(self,input_size,fc1_out,first_width,first_height,kernel_size_list,stride_list,padding_list,outpadding_list,in_channel_list,out_channel_list):
    super().__init__()
    self.fc1 = nn.Linear(input_size,fc1_out)
    self.relu = nn.ReLU()
    self.Deconv_list = []
    for i in range(len(kernel_size_list)):
      self.Deconv_list.append(Convtp_Bn_ReLu(in_channel_list[i],out_channel_list[i],kernel_size_list[i],stride_list[i],
                                             padding_list[i],outpadding_list[i]))
    
    self.Deconvs = nn.Sequential(*self.Deconv_list)

    self.first_width = first_width
    self.first_height = first_height
    
  def forward(self,x):
    x = self.fc1(x)
    x = self.relu(x)
    x = x.view(-1, 1, self.first_height, self.first_width)

    x = self.Deconvs(x)
    return x









#classfication_model
#one-dimensional

#submodule

#使ってない
class Conv1d_Bn_Sigmoid_Pool(nn.Module):
    def __init__(self, inc, outc, ks, pool):
        super().__init__()
        self.CNN1d = nn.Sequential(
            nn.Conv1d(inc, outc, ks),
            nn.BatchNorm1d(outc),
            nn.Sigmoid(),
            nn.MaxPool1d(pool)
        
        )
    def forward(self, x):
        x = self.CNN1d(x)
        return x


class Conv1d_Bn_ReLU_Pool(nn.Module):
    def __init__(self, inc, outc, ksize, pool):
        super().__init__()
        self.CNN1d = nn.Sequential(
            nn.Conv1d(inc, outc, ksize),
            nn.BatchNorm1d(outc),
            nn.ReLU(),
            nn.MaxPool1d(pool)
        
        )
    def forward(self, x):
        x = self.CNN1d(x)
        return x



#model
#sigmoid BCEloss用モデル　学習関数に変更必要　保留中
class cnn1d_sigmoidout(nn.Module):
    """
    chlistの長さだけ一つ長くする
    BCELoss用にsigmoidで出力
    """
    def __init__(self, chlist, kslist, poollist, classes, drop, linear2_in):
        super().__init__()
        #CNNの層数
        length = len(kslist)

        CNNlist = []

        for i in range(length):
            CNNlist.append(Conv1d_Bn_ReLU_Pool(chlist[i], chlist[i + 1], kslist[i], poollist[i]))

        self.CNNS = nn.Sequential(*CNNlist)

        self.flat = nn.Flatten()

        self.fc1 = nn.LazyLinear(linear2_in)
        self.fc2 = nn.Linear(linear2_in, classes)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        self.drop = nn.Dropout(p = drop)

    def forward(self, x):
        x = self.CNNS(x)
        x = self.flat(x)
        x = self.drop(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        #BCELoss用
        return x



class simnet_cnn1d(nn.Module):
    def __init__(self, cnn_layer_num, inc, outc_list, ksize_list, pool_list, classes, drop, linear2_in):
        super().__init__()
        self.cnn_layer_num = cnn_layer_num
        self.CNN1 =  Conv1d_Bn_ReLU_Pool(inc, outc_list[0], ksize_list[0], pool_list[0])

        self.CNNlist = []
        for i in range(1, self.cnn_layer_num):
            self.CNNlist.append(Conv1d_Bn_ReLU_Pool(outc_list[i-1], outc_list[i], ksize_list[i], pool_list[i]))
        self.CNNS = nn.Sequential(*self.CNNlist)

        self.flat = nn.Flatten()

        self.fc1 = nn.LazyLinear(linear2_in)
        self.fc2 = nn.Linear(linear2_in, classes)
        self.relu = nn.ReLU()

        self.drop = nn.Dropout(p = drop)

    def forward(self, x):
        x = self.CNN1(x)
        x = self.CNNS(x)
        x = self.flat(x)
        x = self.drop(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.drop(x)
        x = self.fc2(x)
        #NLLLoss用
        return F.log_softmax(x, dim = 1)



#two-dimensional

#submodule
#畳み込み構造
class Conv_Bn_ReLu(nn.Module):
  def __init__(self,input_channel,output_channel,kernel_size):
    super().__init__()
    self.CNN = nn.Sequential(nn.Conv2d(input_channel, output_channel, kernel_size),
                             nn.BatchNorm2d(output_channel),
                             nn.ReLU(),
                             )
  def forward(self,x):
    x = self.CNN(x)
    return x


class Conv_Bn_ReLu_He_weight(nn.Module):
  """
  He initialization
  """
  def __init__(self,input_channel,output_channel,kernel_size):
    super().__init__()
    self.CNN = nn.Sequential(nn.Conv2d(input_channel, output_channel, kernel_size),
                             nn.BatchNorm2d(output_channel),
                             nn.ReLU(),
                             )
    
    #CNN層の重み初期化
    nn.init.kaiming_uniform_(self.CNN[0].weight, nonlinearity = 'relu')

    #BN層は重み初期化出来ない？
    # nn.init.kaiming_uniform_(self.CNN[1].weight, nonlinearity = 'relu')

  def forward(self,x):
    x = self.CNN(x)
    return x

#LeakyReLU
class Conv_LeakyReLU(nn.Module):
    def __init__(self, inc, outc, ks, strd, pad, negslo):
        super().__init__()
        self.CNN = nn.Sequential(nn.Conv2d(inc, outc, ks, strd, pad),
                                 nn.LeakyReLU(negative_slope = negslo),
                                )
    def forward(self, x):
        x = self.CNN(x)
        return x


class Conv_Bn_LeakyReLU(nn.Module):
    def __init__(self, inc, outc, ks, strd, pad, negslo):
        super().__init__()
        self.CNN = nn.Sequential(nn.Conv2d(inc, outc, ks, strd, pad),
                             nn.BatchNorm2d(outc),
                             nn.LeakyReLU(negative_slope = negslo),
                             )
    def forward(self, x):
        x = self.CNN(x)
        return x


class Conv_Bn_LeakyReLu_He_weight(nn.Module):
  """
  LeakyReLU and He initialization
  Default of LeakyReLU's negative_slope is 0.01.
  """
  def __init__(self, input_channel, output_channel, kernel_size, negslo):
    super().__init__()
    self.CNN = nn.Sequential(nn.Conv2d(input_channel, output_channel, kernel_size),
                             nn.BatchNorm2d(output_channel),
                             nn.LeakyReLU(negative_slope = negslo),
                             )
    
    #CNN層の重み初期化
    nn.init.kaiming_uniform_(self.CNN[0].weight, nonlinearity = 'leaky_relu')

    # #BN層は重み初期化出来ない？
    #nn.init.kaiming_uniform_(self.CNN[1].weight, nonlinearity = 'leaky_relu')

  def forward(self,x):
    x = self.CNN(x)
    return x

#sigmoid
class Conv_Sigmoid(nn.Module):
    def __init__(self, inc, outc, ks, strd, pad):
        super().__init__()
        self.CNN = nn.Sequential(
            nn.Conv2d(inc, outc, ks, strd, pad),
            nn.Sigmoid()
            )
    
    def forward(self, x):
        x = self.CNN(x)
        return x

#プーリング
class Conv_Bn_ReLU_Pool(nn.Module):
    def __init__(self, inc, outc, ksize, pool):
        super().__init__()
        self.CNN = nn.Sequential(
            nn.Conv2d(inc, outc, ksize),
            nn.BatchNorm2d(outc),
            nn.ReLU(),
            nn.MaxPool2d((pool, pool))
            )
        
    def forward(self, x):
        x = self.CNN(x)
        return x




#model
class simnet_cnn_allsize_ver6_size_test(nn.Module):
  """
  This is a test model of linear-layer's input of simnet_cnn_allsize_ver6 .
  """
  def __init__(self, cnn_layer_num, inc, output_channel_list, ksize_list, pool_list):
    super().__init__()
    self.cnn_layer_num = cnn_layer_num
    self.CNN1 = Conv_Bn_ReLU_Pool(inc, output_channel_list[0],ksize_list[0], pool_list[0])
    
    self.CNNlist=[]
    for i in range(1, self.cnn_layer_num):
      self.CNNlist.append(Conv_Bn_ReLU_Pool(output_channel_list[i-1], output_channel_list[i] ,ksize_list[i], pool_list[i]))

    self.CNNS = nn.Sequential(*self.CNNlist)
    self.flat = nn.Flatten()

  def forward(self,x):
    x = self.CNN1(x)
    x = self.CNNS(x)
    x = self.flat(x)
    return x

class simnet_cnn_allsize_ver6(nn.Module):
  """
  This is classfication-model. Last layer of this model is log_softmax, so loss-function should be NLLLoss.
  Average Pooling layer is not used in this model.
  cnn_layer_num=len(outpuut_channel_list)=len(kernel_size_list)
  add Drop out layer
  Pooling
  """
  def __init__(self, cnn_layer_num, inc, inh, inw, outc_list, ksize_list, pool_list, classes, drop, linear2_in):
    super().__init__()
    self.cnn_layer_num = cnn_layer_num

    self.CNN1 = Conv_Bn_ReLU_Pool(inc, outc_list[0], ksize_list[0], pool_list[0])
    
    self.CNNlist=[]
    for i in range(1, self.cnn_layer_num):
      self.CNNlist.append(Conv_Bn_ReLU_Pool(outc_list[i-1], outc_list[i], ksize_list[i], pool_list[i]))
    self.CNNS = nn.Sequential(*self.CNNlist)

    #torch.flattenと異なり,デフォルトでバッチを平滑化しない
    self.flat = nn.Flatten()
    
    #テストモデルでlinearのinputsizeを自動的に求める
    test_model = simnet_cnn_allsize_ver6_size_test(cnn_layer_num, inc, outc_list, ksize_list, pool_list)
    outsize = model_outputsize_test(test_model, [1, inc, inh, inw])
    del test_model

    #outsize=(1, size)なので[1]を取り出す
    self.fc1 = nn.Linear(outsize[1], linear2_in)
    self.fc2 = nn.Linear(linear2_in, classes)
    self.relu = nn.ReLU()

    del outsize

    self.drop = nn.Dropout(p = drop)


  def forward(self,x):
    x = self.CNN1(x)
    x = self.CNNS(x)
    x = self.flat(x)
    x = self.drop(x)
    x = self.fc1(x)
    x = self.relu(x)
    x = self.drop(x)
    x = self.fc2(x)
    return F.log_softmax(x, dim=1)







class simnet_cnn_allsize_ver5_size_test(nn.Module):
  """
  This is a test model of linear-layer's input of simnet_cnn_allsize_ver5 .
  """
  def __init__(self, cnn_layer_num, input_channel, output_channel_list, kernel_size_list, classes, negslo):
    super().__init__()
    self.cnn_layer_num = cnn_layer_num
    self.CNN1 = Conv_Bn_LeakyReLu_He_weight(input_channel, output_channel_list[0], kernel_size_list[0], negslo)
    
    self.CNNlist=[]
    for i in range(1,self.cnn_layer_num):
      self.CNNlist.append(Conv_Bn_LeakyReLu_He_weight(output_channel_list[i-1], output_channel_list[i], kernel_size_list[i], negslo))

    self.CNNS = nn.Sequential(*self.CNNlist)
    self.flat = nn.Flatten()

  def forward(self,x):
    x = self.CNN1(x)
    x = self.CNNS(x)
    x = self.flat(x)
    return x

class simnet_cnn_allsize_ver5(nn.Module):
  """
  This is classfication-model. Last layer of this model is log_softmax, so loss-function should be NLLLoss.
  Average Pooling layer is not used in this model.
  cnn_layer_num=len(outpuut_channel_list)=len(kernel_size_list)
  He initialization
  add Drop out layer
  """
  def __init__(self, cnn_layer_num, input_channel, in_height, in_width, output_channel_list, kernel_size_list, classes, drop, linear2_in, negslo):
    super().__init__()
    self.cnn_layer_num = cnn_layer_num

    self.CNN1 = Conv_Bn_LeakyReLu_He_weight(input_channel, output_channel_list[0], kernel_size_list[0], negslo)
    
    self.CNNlist=[]
    for i in range(1,self.cnn_layer_num):
      self.CNNlist.append(Conv_Bn_LeakyReLu_He_weight(output_channel_list[i-1], output_channel_list[i], kernel_size_list[i], negslo))
    self.CNNS = nn.Sequential(*self.CNNlist)

    #torch.flattenと異なり,デフォルトでバッチを平滑化しない
    self.flat = nn.Flatten()
    
    #テストモデルでlinearのinputsizeを自動的に求める
    test_model = simnet_cnn_allsize_ver5_size_test(cnn_layer_num, input_channel, output_channel_list\
                                                   , kernel_size_list, classes, negslo)
    outsize = model_outputsize_test(test_model, [1, input_channel, in_height, in_width])
    del test_model

    #outsize=(1, size)なので[1]を取り出す
    self.fc1 = nn.Linear(outsize[1], linear2_in)
    self.fc2 = nn.Linear(linear2_in, classes)
    self.relu = nn.LeakyReLU()

    del outsize

    #重みを初期化
    nn.init.kaiming_uniform_(self.fc1.weight, mode = 'fan_in', nonlinearity='leaky_relu')
    nn.init.kaiming_uniform_(self.fc2.weight, mode = 'fan_in', nonlinearity='leaky_relu')

    self.drop = nn.Dropout(p = drop)


  def forward(self,x):
    x = self.CNN1(x)
    x = self.CNNS(x)
    x = self.flat(x)
    x = self.drop(x)
    x = self.fc1(x)
    x = self.relu(x)
    x = self.drop(x)
    x = self.fc2(x)
    return F.log_softmax(x, dim=1)



#出力にlog_softmax使ってNllLOSSで計算した方が高精度っぽい
#理由はbackward時にlog_softmaxの計算も含めて、パラメータ更新するから？

class simnet_cnn_allsize_ver4_size_test(nn.Module):
  """
  This is a test model of linear-layer's input of simnet_cnn_allsize_ver4 .
  """
  def __init__(self,cnn_layer_num,input_channel,output_channel_list,kernel_size_list,classes):
    super().__init__()
    self.cnn_layer_num = cnn_layer_num
    self.CNN1 = Conv_Bn_ReLu_He_weight(input_channel,output_channel_list[0],kernel_size_list[0])
    
    self.CNNlist=[]
    for i in range(1,self.cnn_layer_num):
      self.CNNlist.append(Conv_Bn_ReLu_He_weight(output_channel_list[i-1],output_channel_list[i],kernel_size_list[i]))

    self.CNNS = nn.Sequential(*self.CNNlist)
    self.flat = nn.Flatten()

  def forward(self,x):
    x = self.CNN1(x)
    x = self.CNNS(x)
    x = self.flat(x)
    return x

class simnet_cnn_allsize_ver4(nn.Module):
  """
  This is classfication-model. Last layer of this model is log_softmax, so loss-function should be NLLLoss.
  Average Pooling layer is not used in this model.
  cnn_layer_num=len(outpuut_channel_list)=len(kernel_size_list)
  He initialization
  add Drop out layer
  """
  def __init__(self, cnn_layer_num, input_channel, in_height, in_width, output_channel_list, kernel_size_list, classes, drop, linear2_in):
    super().__init__()
    self.cnn_layer_num = cnn_layer_num

    self.CNN1 = Conv_Bn_ReLu_He_weight(input_channel, output_channel_list[0], kernel_size_list[0])
    
    self.CNNlist=[]
    for i in range(1,self.cnn_layer_num):
      self.CNNlist.append(Conv_Bn_ReLu_He_weight(output_channel_list[i-1], output_channel_list[i], kernel_size_list[i]))
    self.CNNS = nn.Sequential(*self.CNNlist)

    #torch.flattenと異なり,デフォルトでバッチを平滑化しない
    self.flat = nn.Flatten()
    
    #テストモデルでlinearのinputsizeを自動的に求める
    test_model = simnet_cnn_allsize_ver4_size_test(cnn_layer_num, input_channel, output_channel_list\
                                                   , kernel_size_list, classes)
    outsize = model_outputsize_test(test_model, [1, input_channel, in_height, in_width])
    del test_model

    #outsize=(1, size)なので[1]を取り出す
    self.fc1 = nn.Linear(outsize[1], linear2_in)
    self.fc2 = nn.Linear(linear2_in, classes)
    self.relu = nn.ReLU()

    del outsize

    #重みを初期化
    nn.init.kaiming_uniform_(self.fc1.weight, mode = 'fan_in', nonlinearity='relu')
    nn.init.kaiming_uniform_(self.fc2.weight, mode = 'fan_in', nonlinearity='relu')

    self.drop = nn.Dropout(p = drop)


  def forward(self,x):
    x = self.CNN1(x)
    x = self.CNNS(x)
    x = self.flat(x)
    x = self.drop(x)
    x = self.fc1(x)
    x = self.relu(x)
    x = self.drop(x)
    x = self.fc2(x)
    return F.log_softmax(x, dim=1)


class simnet_cnn_allsize_ver3_size_test(nn.Module):
  """
  This is a test model of linear-layer's input of simnet_cnn_allsize_ver3 .
  """
  def __init__(self,cnn_layer_num,input_channel,output_channel_list,kernel_size_list,classes):
    super().__init__()
    self.cnn_layer_num = cnn_layer_num
    self.CNN1 = Conv_Bn_ReLu(input_channel,output_channel_list[0],kernel_size_list[0])
    self.CNNlist=[]
    for i in range(1,self.cnn_layer_num):
      self.CNNlist.append(Conv_Bn_ReLu(output_channel_list[i-1],output_channel_list[i],kernel_size_list[i]))

    self.CNNS = nn.Sequential(*self.CNNlist)
    self.flat = nn.Flatten()

  def forward(self,x):
    x = self.CNN1(x)
    x = self.CNNS(x)
    #x = torch.flatten(x, 1)
    x = self.flat(x)
    return x

class simnet_cnn_allsize_ver3(nn.Module):
  """
  This is classfication-model. Last layer of this model is log_softmax, so loss-function should be NLLLoss.
  Average Pooling layer is not used in this model.
  cnn_layer_num=len(outpuut_channel_list)=len(kernel_size_list)
  """
  def __init__(self, cnn_layer_num, input_channel, in_height, in_width, output_channel_list, kernel_size_list, classes):
    super().__init__()
    self.cnn_layer_num = cnn_layer_num
    self.CNN1 = Conv_Bn_ReLu(input_channel,output_channel_list[0],kernel_size_list[0])
    self.CNNlist=[]
    for i in range(1,self.cnn_layer_num):
      self.CNNlist.append(Conv_Bn_ReLu(output_channel_list[i-1],output_channel_list[i],kernel_size_list[i]))

    self.CNNS = nn.Sequential(*self.CNNlist)

    #torch.flattenと異なり,デフォルトでバッチを平滑化しない
    self.flat = nn.Flatten()
    
    #テストモデルでlinearのinputsizeを自動的に求める
    test_model = simnet_cnn_allsize_ver3_size_test(cnn_layer_num, input_channel, output_channel_list\
                                                   , kernel_size_list, classes)
    
    outsize = model_outputsize_test(test_model, [1, input_channel, in_height, in_width])
    del test_model
    #outsize=(1, size)なので[1]を取り出す
    self.fc = nn.Linear(outsize[1], classes)
    del outsize

  def forward(self,x):
    x = self.CNN1(x)
    x = self.CNNS(x)
    x = self.flat(x)
    x = self.fc(x)
    return F.log_softmax(x, dim=1)

#モデルをテストして全結合の入力を求める
def model_outputsize_test(model, input_size):
  """
  input_size : tuple or list
  """
  input = torch.randn(input_size)
  # input = input.to(device)
  model.eval()
  output = model(input)
  return output.shape





class simnet_cnn_allsize_ver2(nn.Module):
  """
  This is classfication-model. Last layer of this model is log_softmax, so loss-function should be NLLLoss.
  cnn_layer_num=len(outpuut_channel_list)=len(kernel_size_list)
  """
  def __init__(self,cnn_layer_num,input_channel,output_channel_list,kernel_size_list,classes):
    super().__init__()
    self.cnn_layer_num = cnn_layer_num
    self.CNN1 = Conv_Bn_ReLu(input_channel,output_channel_list[0],kernel_size_list[0])
    self.CNNlist=[]
    for i in range(1,self.cnn_layer_num):
      self.CNNlist.append(Conv_Bn_ReLu(output_channel_list[i-1],output_channel_list[i],kernel_size_list[i]))

    self.CNNS = nn.Sequential(*self.CNNlist)
    self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1,1))
    self.fc = nn.Linear(output_channel_list[self.cnn_layer_num-1],classes)
  def forward(self,x):
    x = self.CNN1(x)
    x = self.CNNS(x)
    x = self.avgpool(x)
    #第2引数で1次元以降を平滑化している、0次元のバッチは平滑化していない、デフォルトでは0次元から平滑化
    x = torch.flatten(x, 1)
    x = self.fc(x)
    return F.log_softmax(x, dim=1)
    #return x

class simnet_cnn_allsize(nn.Module):
  """
  This is classfication-model. Last layer of this model is log_softmax, so loss-function should be NLLLoss.
  """
  def __init__(self,input_channel,output_channel_list,kernel_size_list,classes):
    super().__init__()
    self.CNN1 = Conv_Bn_ReLu(input_channel,output_channel_list[0],kernel_size_list[0])
    self.CNN2 = Conv_Bn_ReLu(output_channel_list[0],output_channel_list[1],kernel_size_list[1])
    self.CNN3 = Conv_Bn_ReLu(output_channel_list[1],output_channel_list[2],kernel_size_list[2])
    self.CNN4 = Conv_Bn_ReLu(output_channel_list[2],output_channel_list[3],kernel_size_list[3])
    self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1,1))
    self.fc = nn.Linear(output_channel_list[3],classes)
  def forward(self,x):
    x = self.CNN1(x)
    x = self.CNN2(x)
    x = self.CNN3(x)
    x = self.CNN4(x)
    x = self.avgpool(x)
    #第2引数で1次元以降を平滑化している、0次元のバッチは平滑化していない、デフォルトでは0次元から平滑化
    x = torch.flatten(x, 1)
    x = self.fc(x)
    return F.log_softmax(x, dim=1)
    #return x

class simnet_cnn_128(nn.Module):
    """
    This is classfication-model. Last layer of this model is log_softmax, so loss-function should be NLLLoss.
    """
    def __init__(self,classes):
        super().__init__()
        self.CNN1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=4),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            #nn.MaxPool2d(2)
            )
        self.CNN2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            #nn.MaxPool2d(2)
            )
        self.CNN3 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            #nn.MaxPool2d(2)
            )
        self.CNN4 = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            #nn.MaxPool2d(2)
            )
        self.dropout = nn.Dropout(0.1)
        self.l1 = nn.Linear(9 * 1 * 16, 128)
        self.l2 = nn.Linear(512, 256)
        self.l3 = nn.Linear(256, 128)
        self.l4 = nn.Linear(128,128)
        self.last = nn.Linear(128, classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.CNN1(x)
        x = self.CNN2(x)
        x = self.CNN3(x)
        x = self.CNN4(x)
        x = x.view(x.size(0), -1)
        x = self.l1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.l4(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.l4(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.l4(x)
        x = self.relu(x)
        x = self.last(x)
        return F.log_softmax(x, dim=1)
        #return x

class simnet_cnn_256(nn.Module):
    """
    This is classfication-model. Last layer of this model is log_softmax, so loss-function should be NLLLoss.
    """
    def __init__(self,classes):
        super().__init__()
        self.CNN1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=4),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            #nn.MaxPool2d(2)
            )
        self.CNN2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            #nn.MaxPool2d(2)
            )
        self.CNN3 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            #nn.MaxPool2d(2)
            )
        self.CNN4 = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            #nn.MaxPool2d(2)
            )
        self.dropout = nn.Dropout(0.1)
        self.l1 = nn.Linear(9 * 9 * 16, 512)
        self.l2 = nn.Linear(512, 256)
        self.l3 = nn.Linear(256, 128)
        self.l4 = nn.Linear(128,128)
        self.last = nn.Linear(128, classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.CNN1(x)
        x = self.CNN2(x)
        x = self.CNN3(x)
        x = self.CNN4(x)
        x = x.view(x.size(0), -1)
        x = self.l1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.l2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.l3(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.l4(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.l4(x)
        x = self.relu(x)
        x = self.last(x)
        return F.log_softmax(x, dim=1)
        #return x

class simnet_cnn_500(nn.Module):
    """
    This is classfication-model. Last layer of this model is log_softmax, so loss-function should be NLLLoss.
    """
    def __init__(self,classes):
        super().__init__()
        self.CNN1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=4),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            #nn.MaxPool2d(2)
            )
        self.CNN2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            #nn.MaxPool2d(2)
            )
        self.CNN3 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            #nn.MaxPool2d(2)
            )
        self.CNN4 = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=3),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            #nn.MaxPool2d(2)
            )
        self.dropout = nn.Dropout(0.1)
        self.l1 = nn.Linear(16 * 16 * 11, 1280)
        self.l2 = nn.Linear(1280, 512)
        self.l3 = nn.Linear(512, 256)
        self.l4 = nn.Linear(256,128)
        self.last = nn.Linear(128, classes)
        self.relu = nn.ReLU()
    def forward(self, x):
        x = self.CNN1(x)
        x = self.CNN2(x)
        x = self.CNN3(x)
        x = self.CNN4(x)
        x = x.view(x.size(0), -1)
        x = self.l1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.l2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.l3(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.l4(x)
        x = self.relu(x)
        x = self.last(x)
        return F.log_softmax(x, dim=1)
        #return x

class simnet_linear_allsize(nn.Module):
    """
    This is classfication-model. Last layer of this model is log_softmax, so loss-function should be NLLLoss.
    """
    def __init__(self,laynum,classes,input_size,drop):
        super().__init__()
        self.l1 = nn.Linear(input_size, input_size)
        self.l2 = nn.Linear(input_size, classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(drop)
        self.laynum = laynum

    def forward(self, x):
      x = self.l1(x)
      x = self.relu(x)
      for i in range(self.laynum):
        x = self.l1(x)
        x = self.relu(x)
        x = self.dropout(x)
      x = self.l1(x)
      x = self.l2(x)
      return F.log_softmax(x, dim=1)
      #return x






#nonsim
class in_ver5(nn.Module):
    def __init__(self,classes):
        super().__init__()
        self.CNN1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=16),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.CNN2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=4),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.CNN3 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=4),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.CNN4 = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=4),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.dropout = nn.Dropout(0.1)
        self.l1 = nn.Linear(16 * 12 * 12, 1024)
        self.l2 = nn.Linear(2048, 1024)
        self.l3 = nn.Linear(1024, 512)
        self.l4 = nn.Linear(512, 128)
        self.fc = nn.Linear(128, classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.CNN1(x)
        x = self.CNN2(x)
        x = self.CNN3(x)
        x = self.CNN4(x)
        x = x.view(x.size(0), -1)
        x = self.l1(x)
        x = self.relu(x)
        x = self.dropout(x)
        # x = self.l2(x)
        # x = self.relu(x)
        # x = self.dropout(x)
        x = self.l3(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.l4(x)
        x = self.relu(x)
        x = self.fc(x)
        return F.log_softmax(x, dim=1)

class mnist_net(nn.Module):
    def __init__(self,classes):
        super().__init__()
        self.CNN1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=4),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.CNN2 = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=4),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2))
  
        self.dropout = nn.Dropout(0.1)
        self.l1 = nn.Linear(58 * 58 * 64, 1024)
        self.l2 = nn.Linear(2048, 1024)
        self.l3 = nn.Linear(1024, 512)
        self.l4 = nn.Linear(512, 128)
        self.l5 = nn.Linear(16*4*4, 128)
        self.l6 = nn.Linear(128,64)
        self.l7 = nn.Linear(64,32)
        self.fc = nn.Linear(32, classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.CNN1(x)
        x = self.CNN2(x)
        x = x.view(x.size(0), -1)
        x = self.l5(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.l6(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.l7(x)
        x = self.relu(x)
        x = self.fc(x)
        return F.log_softmax(x, dim=1)

class cifar_net_g(nn.Module):
    def __init__(self,classes):
        super().__init__()
        self.CNN1 = nn.Sequential(
            nn.Conv2d(1, 128, kernel_size=3),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            # nn.MaxPool2d(2)
            )
        self.CNN2 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # nn.MaxPool2d(2)
            )
        self.CNN3 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2)
            )
        self.CNN4 = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=3),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2)
            )
        self.dropout = nn.Dropout(0.1)
        self.l1 = nn.Linear(58 * 58 * 64, 1024)
        self.l2 = nn.Linear(2048, 1024)
        self.l3 = nn.Linear(1024, 512)
        self.l4 = nn.Linear(512, 128)
        self.l5 = nn.Linear(16*5*5, 128)
        self.l6 = nn.Linear(128,64)
        self.l7 = nn.Linear(64,32)
        self.l8 = nn.Linear(128,128)
        self.fc = nn.Linear(128, classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.CNN1(x)
        x = self.CNN2(x)
        x = self.CNN3(x)
        x = self.CNN4(x)
        x = x.view(x.size(0), -1)
        x = self.l5(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.l8(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.l8(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.l8(x)
        x = self.relu(x)
        x = self.fc(x)
        return F.log_softmax(x, dim=1)

class cifar_net_c(nn.Module):
    def __init__(self,classes):
        super().__init__()
        self.CNN1 = nn.Sequential(
            nn.Conv2d(3, 128, kernel_size=3),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            # nn.MaxPool2d(2)
            )
        self.CNN2 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # nn.MaxPool2d(2)
            )
        self.CNN3 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2)
            )
        self.CNN4 = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=3),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2)
            )
        self.dropout = nn.Dropout(0.1)
        self.l1 = nn.Linear(58 * 58 * 64, 1024)
        self.l2 = nn.Linear(2048, 1024)
        self.l3 = nn.Linear(1024, 512)
        self.l4 = nn.Linear(512, 128)
        self.l5 = nn.Linear(16*5*5, 128)
        self.l6 = nn.Linear(128,64)
        self.l7 = nn.Linear(64,32)
        self.l8 = nn.Linear(128,128)
        self.fc = nn.Linear(128, classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.CNN1(x)
        x = self.CNN2(x)
        x = self.CNN3(x)
        x = self.CNN4(x)
        x = x.view(x.size(0), -1)
        x = self.l5(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.l8(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.l8(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.l8(x)
        x = self.relu(x)
        x = self.fc(x)
        return F.log_softmax(x, dim=1)







#Resnet
def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels,out_channels,kernel_size=3,stride=stride,padding=1,bias=False,)

def conv1x1(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1  # 出力のチャンネル数を入力のチャンネル数の何倍に拡大するか

    def __init__(self,in_channels,channels,stride=1):
        super().__init__()
        self.conv1 = conv3x3(in_channels, channels, stride)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(channels, channels)
        self.bn2 = nn.BatchNorm2d(channels)

        # 入力と出力のチャンネル数が異なる場合、x をダウンサンプリングする。
        if in_channels != channels * self.expansion:
            self.shortcut = nn.Sequential(
                conv1x1(in_channels, channels * self.expansion, stride),
                nn.BatchNorm2d(channels * self.expansion),
            )
        else:
            self.shortcut = nn.Sequential()

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += self.shortcut(x)

        out = self.relu(out)

        return out
    
class Bottleneck(nn.Module):
    expansion = 4  # 出力のチャンネル数を入力のチャンネル数の何倍に拡大するか

    def __init__(self, in_channels, channels, stride=1):
        print(in_channels, out_channels)
        super().__init__()
        self.conv1 = conv1x1(in_channels, channels)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = conv3x3(channels, channels, stride)
        self.bn2 = nn.BatchNorm2d(channels)
        self.conv3 = conv1x1(channels, channels * self.expansion)
        self.bn3 = nn.BatchNorm2d(channels * self.expansion)
        self.relu = nn.ReLU(inplace=True)

        # 入力と出力のチャンネル数が異なる場合、x をダウンサンプリングする。
        if in_channels != channels * self.expansion:
            self.shortcut = nn.Sequential(
                conv1x1(in_channels, channels * self.expansion, stride),
                nn.BatchNorm2d(channels * self.expansion),
            )
        else:
            self.shortcut = nn.Sequential()

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += self.shortcut(x)

        out = self.relu(out)

        return out
    
class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000):
        super().__init__()

        self.in_channels = 64
        self.conv1 = nn.Conv2d(
            3, self.in_channels, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        # 重みを初期化する。
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, channels, blocks, stride):
        layers = []

        # 最初の Residual Block
        layers.append(block(self.in_channels, channels, stride))

        # 残りの Residual Block
        self.in_channels = channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x
    
def resnet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])

def resnet34():
    return ResNet(BasicBlock, [3, 4, 6, 3])

def resnet50():
    return ResNet(Bottleneck, [3, 4, 6, 3])

def resnet101():
    return ResNet(Bottleneck, [3, 4, 23, 3])

def resnet152():
    return ResNet(Bottleneck, [3, 8, 36, 3])
