import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader,Dataset
import matplotlib.pyplot as plt
import time

#データセットクラス
class MyDataset(Dataset):
  def __init__(self, xdata, ydata):
    self.data = xdata
    self.label = ydata
  def __len__(self):
    return len(self.label)
  def __getitem__(self, idx):
    x = self.data[idx]
    y = self.label[idx]
    return x, y

#データローダで使う
def my_collate_fn(batch):
  xdata, ydata = list(zip(*batch))
  xs = list(xdata)
  ys = list(ydata)
  return xs, ys



#NMT
#バッチパディング関数
def batch_pad_nmt(xs, ys, label_pad_value, device):
  xs1, ys1, ys2 = [], [], []
  for i in range(len(xs)):
    ids = xs[i]
    #原文の先頭<s>以外をxs1に
    xs1.append(torch.LongTensor(ids[1:]))

    ids = ys[i]
    #訳文の文末</s>以外をys1に
    ys1.append(torch.LongTensor(ids[:-1]))
    #訳文の先頭<s>以外をys2に
    ys2.append(torch.LongTensor(ids[1:]))
  
  xs1 = pad_sequence(xs1, batch_first = True).to(device)
  ys1 = pad_sequence(ys1, batch_first = True).to(device)
  gans = pad_sequence(ys2, batch_first = True, padding_value = label_pad_value).to(device)
  return xs1,ys1,gans

#nmtモデルの学習関数
def train_nmt_model(model, epochs, optimizer, lossfunc, device, dl):
  
  t1 = time.time()
  loss_list = []
  
  for epoch in range(epochs):
    model.train()
    loss_print = 0
    data_num = 0

    for xs, ys in dl:
      #全単語数加算
      for i in range(len(xs)):
        data_num += len(xs[i])
      
      #バッチをパディング
      jinput,einput,gans = batch_pad_nmt(xs, ys, -1, device)

      output = model(jinput, einput)

      #エポック毎の損失を加算
      loss = lossfunc(output[0], gans[0])
      for k in range(1,len(gans)):
        loss += lossfunc(output[k], gans[k])
      loss_print += loss.item()

      #更新
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
    
    loss_list.append(loss_print/data_num)
    print('---------------------------------')
    print(f'エポック{epoch+1} ,train_loss: {loss_print/data_num}')
    
    t2=time.time()
    caltime=(t2-t1)/60
    print(f'time:{caltime}分')
    t1=time.time()
  
  return loss_list










#LSTM
#バッチパディング関数
def batch_pad(xs, ys, label_pad_value, device):
  xs1, ys1 = [], []
  for i in range(len(xs)):
    ids = xs[i]
    xs1.append(torch.LongTensor(ids))
    ids = ys[i]
    ys1.append(torch.LongTensor(ids))
  xsl = pad_sequence(xs1, batch_first = True).to(device)
  ysl = pad_sequence(ys1, batch_first = True, padding_value = label_pad_value).to(device)
  #ysl = pad_sequence(ys1, batch_first = True, padding_value = label_pad_value)

  return xsl, ysl

#lstm学習関数ver2:メモリ使いすぎてるかも
def train_lstm_model_ver2(model, epochs, optimizer, lossfunc, device, dl, tl = None):
  """
  tl is arbitrary.
  """
  # if loaders.get('tl') != None:
  #   dataloader = loaders['tl']
  # if loaders.get('vl') != None:
  #   testdataloader = loaders['vl']

  t1 = time.time()
  loss_list = []
  acc_list = []
  loss_val_list = []
  acc_val_list = []
  for epoch in range(epochs):
    model.train()
    loss_print = 0
    correct = 0
    data_num = 0

    for xs, ys in dl:
      #全単語数加算
      for i in range(len(xs)):
        data_num += len(xs[i])

      #バッチをパディング
      xsl, ysl = batch_pad(xs, ys, -1, device)

      output = model(xsl)
      # ysl = ysl.type(torch.LongTensor).to(device)

      #損失
      loss = lossfunc(output[0], ysl[0])
      for k in range(1,len(ysl)):
        loss += lossfunc(output[k],ysl[k])
      loss_print += loss.item()

      #正答率
      ans = torch.argmax(output, dim = 2)
      correct += torch.sum(ans == ysl).item()

      #更新
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

    
    #エポック毎の損失と正答率
    loss_list.append(loss_print/data_num)
    acc_list.append(correct/data_num)
    print('---------------------------------')
    print(f'エポック{epoch+1} ,train_loss: {loss_print/data_num} ,train_acc: {correct/data_num}')

    #評価用データローダがあるなら評価
    if tl != None:
      loss_val, acc_val = val_lstm_model(model, tl, lossfunc, device)
      acc_val_list.append(acc_val)
      loss_val_list.append(loss_val)
      print(f'val_loss: {loss_val} ,val_acc: {acc_val}')
    
    t2=time.time()
    caltime=(t2-t1)/60
    print(f'time:{caltime}分')
    t1=time.time()
  
  return loss_list, loss_val_list, acc_list, acc_val_list

#lstm学習関数
def train_lstm_model(model, epochs, dataloader, testdataloader, optimizer, lossfunc, device):
  t1 = time.time()
  loss_list = []
  acc_list = []
  loss_val_list = []
  acc_val_list = []
  for epoch in range(epochs):
    model.train()
    loss_print = 0
    correct = 0
    data_num = 0

    for xs, ys in dataloader:
      #全単語数加算
      for i in range(len(xs)):
        data_num += len(xs[i])

      #バッチをパディング
      xsl, ysl = batch_pad(xs, ys, -1, device)

      output = model(xsl)
      ysl = ysl.type(torch.LongTensor).to(device)

      #損失
      loss = lossfunc(output[0], ysl[0])
      for k in range(1,len(ysl)):
        loss += lossfunc(output[k],ysl[k])
      loss_print += loss.item()

      #正答率
      ans = torch.argmax(output, dim = 2)
      correct += torch.sum(ans == ysl).item()

      #更新
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

    
    #エポック毎の損失と正答率
    loss_list.append(loss_print/data_num)
    acc_list.append(correct/data_num)
    print(f'エポック{epoch+1} ,train_loss: {loss_print/data_num} ,train_acc: {correct/data_num}')

    #評価用データローダがあるなら評価
    if (testdataloader != None):
      loss_val, acc_val = val_lstm_model(model, testdataloader, lossfunc, device)
      acc_val_list.append(acc_val)
      loss_val_list.append(loss_val)
      print(f'-----------------val_loss: {loss_val} ,val_acc: {acc_val}')
    
    t2=time.time()
    caltime=(t2-t1)/60
    print(f'time:{caltime}分')
    t1=time.time()
  
  return loss_list, loss_val_list, acc_list, acc_val_list

#lstmテスト関数
def val_lstm_model(model, dataloader, lossfunc, device):
  model.eval()
  data_num = 0
  correct = 0
  loss_print = 0
  with torch.no_grad():  
    for xs, ys in dataloader:
      #全単語数加算
      for i in range(len(xs)):
        data_num += len(xs[i])
      xsl, ysl = batch_pad(xs, ys, -1, device)

      output = model(xsl)
      ysl = ysl.type(torch.LongTensor).to(device)

      #損失
      loss = lossfunc(output[0], ysl[0])
      for k in range(1,len(ysl)):
        loss += lossfunc(output[k],ysl[k])
      loss_print += loss.item()

      #正答率
      ans = torch.argmax(output, dim = 2)
      correct += torch.sum(ans == ysl).item()
    
    print(f'val_loss:{loss_print/data_num}, val_correct_rate:{correct/data_num}')

  return loss_print/data_num, correct/data_num





#学習曲線表示
#Displaying learning-curv_ver2
def learning_curv_ver2( fig_w, fig_h, labelfontsize, scalefontsize, tls = None, vls = None, tas = None, vas = None):
  """
  you must prepare vls.
  """
  rcparams_dic = {
    'figure.figsize': (fig_w,fig_h),
    'axes.labelsize': labelfontsize,
    'xtick.labelsize': scalefontsize,
    'ytick.labelsize': scalefontsize,
  }
  plt.rcParams.update(rcparams_dic)

  #正解率の配列を持っている場合
  if (tas != None) or (vas != None):
    plt.subplots_adjust(wspace=0.4, hspace=0.6)
    plt.subplot(1,2,1)
  
  if tls != None:
    plt.plot(range(1, 1 + len(tls)), tls, label="training")
  plt.plot(range(1, 1 + len(vls)), vls, label="validation")
  plt.xlabel('Epochs')
  plt.ylabel('Loss')
  plt.legend()

  if (vas != None) or (tas != None):
    plt.subplot(1,2,2)
    if tas != None:
      plt.plot(range(1, 1 + len(tas)), tas, label="training")
    plt.plot(range(1, 1 + len(vas)), vas, label="validation")
    plt.xlabel('Epochs')
    plt.ylabel('Acc')
    plt.legend()

  plt.show()

#Displaying learning-curv
def learning_curv(acc_cf, valloss, valacc, train_cf, trainloss, trainacc, fig_w, fig_h, labelfontsize, scalefontsize):
  
  rcparams_dic = {
    'figure.figsize': (fig_w,fig_h),
    'axes.labelsize': labelfontsize,
    'xtick.labelsize': scalefontsize,
    'ytick.labelsize': scalefontsize,
  }
  plt.rcParams.update(rcparams_dic)


  if acc_cf == True:
    plt.subplots_adjust(wspace=0.4, hspace=0.6)
    plt.subplot(1,2,1)
  
  if train_cf == True:
    plt.plot(range(1, 1 + len(trainloss)), trainloss, label="training")
  plt.plot(range(1, 1 + len(valloss)), valloss, label="validation")
  plt.xlabel('Epochs')
  plt.ylabel('Loss')
  plt.legend()

  if acc_cf == True:
    plt.subplot(1,2,2)
    if train_cf == True:
      plt.plot(range(1, 1 + len(trainloss)), trainacc, label="training")
    plt.plot(range(1, 1 + len(trainloss)), valacc, label="validation")
    plt.xlabel('Epochs')
    plt.ylabel('Acc')
    plt.legend()

  plt.show()



# #Displaying learning-curv_ver2
# def learning_curv_ver2( fig_w, fig_h, labelfontsize, scalefontsize, **tl_vl_ta_va):
#   """
#   you must prepare vl and va.
#   tl and ta is optional.
#   listname must be vl, va, tl or ta only.
#   """
#   rcparams_dic = {
#     'figure.figsize': (fig_w,fig_h),
#     'axes.labelsize': labelfontsize,
#     'xtick.labelsize': scalefontsize,
#     'ytick.labelsize': scalefontsize,
#   }
#   plt.rcParams.update(rcparams_dic)

#   if tl_vl_ta_va.get('tl') != None:
#     tls = tl_vl_ta_va['tl']
  
#   if tl_vl_ta_va.get('vl') != None:
#     vls = tl_vl_ta_va['vl']
  
#   if tl_vl_ta_va.get('ta') != None:
#     tas = tl_vl_ta_va['ta']
  
#   if tl_vl_ta_va.get('va') != None:
#     vas = tl_vl_ta_va['va']

#   #正解率の配列を持っている場合
#   if ((tl_vl_ta_va.get('va') != None) or (tl_vl_ta_va.get('ta') != None)):
#     plt.subplots_adjust(wspace=0.4, hspace=0.6)
#     plt.subplot(1,2,1)
  
#   if tl_vl_ta_va.get('tl') != None:
#     plt.plot(range(1, 1 + len(tls)), tls, label="training")
#   plt.plot(range(1, 1 + len(vls)), vls, label="validation")
#   plt.xlabel('Epochs')
#   plt.ylabel('Loss')
#   plt.legend()

#   if ((tl_vl_ta_va.get('va') != None) or (tl_vl_ta_va.get('ta') != None)):
#     plt.subplot(1,2,2)
#     if tl_vl_ta_va.get('ta') != None:
#       plt.plot(range(1, 1 + len(tas)), tas, label="training")
#     plt.plot(range(1, 1 + len(vas)), vas, label="validation")
#     plt.xlabel('Epochs')
#     plt.ylabel('Acc')
#     plt.legend()

#   plt.show()