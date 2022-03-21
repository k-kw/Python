import matplotlib.pyplot as plt
from torch.autograd import Variable
import time
import torch
import numpy as np


def mixup_data(x, y, device, alpha=1.0):

    '''Compute the mixup data. Return mixed inputs, pairs of targets, and lambda'''
    if alpha > 0.:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(device)
    # if use_cuda:
    #     index = torch.randperm(batch_size).cuda()
    # else:
    #     index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index,:]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(y_a, y_b, lam):
    return lambda criterion, pred: lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


#深層学習前の標準化
def tensor_norm_DL(tensor, mean = None, std = None):
  if mean == None:
    mean = torch.mean(tensor)
  if std == None:
    std = torch.mean(tensor)
  re_tensor = (tensor - mean)/std
  return re_tensor, mean, std

#正規化
def tensor_norm(tensor, max = None, min = None):
  if max == None:
     max = torch.max(tensor)
  if min == None:
    min = torch.min(tensor)
  re_tensor = (tensor - min)/(max - min)
  return re_tensor, max, min


#テンソルを分割、標準化、正規化、データセット化
def split_and_dataset_ver2(data, labels, train_len, norm_true_data, norm_true_label):
  #分割
  data_t, data_v = data[:train_len], data[train_len:]
  label_t,label_v = labels[:train_len], labels[train_len:]

  #標準化
  if (norm_true_data == True):
    data_t, mean, std = tensor_norm_DL(data_t)
    data_v, _, _ = tensor_norm_DL(data_v, mean, std)

  #正規化
  if (norm_true_label == True):
    label_t, max, min = tensor_norm(label_t)
    label_v, _, _ = tensor_norm(label_v, max, min)

  #データセット化
  dataset_train = torch.utils.data.TensorDataset(data_t, label_t)
  dataset_val = torch.utils.data.TensorDataset(data_v, label_v)
  return dataset_train, dataset_val


#データを分割してデータセット化
def split_and_dataset(data, labels, train_len):
  data_t, data_v = data[:train_len], data[train_len:]
  label_t,label_v = labels[:train_len], labels[train_len:]
  dataset_train = torch.utils.data.TensorDataset(data_t, label_t)
  dataset_val = torch.utils.data.TensorDataset(data_v, label_v)
  return dataset_train, dataset_val


#Make dataset from sindatpath and bindatpath(for decoder model)
def mysimbin_to_dataset_ver2(simdat_path, bindat_path, sizex_sim, sizex_bin, sizey_bin, num, sim_byte_num, \
  bin_byte_num, separate_true, train_length, normalize_true = False, label_img_norm_true = False):
  """
  Convert one-dimmensional simdat and two-dimmensional simdat into torch dataset(train and val)
  separate_true:bool (confirm separating dataset into train and val)
  normalize_true: bool   If this is True, sim-data's average and its std are zero and one.
  label_img_norm_true: bool   If this is True, label-images's value are convert into value from zero to one.
  """
  import py_func.dat_rb_func as drb

  simdat = drb.sim_label_read(simdat_path, sizex_sim, num, False, sim_byte_num)
  bindat = drb.bin_read(bindat_path, bin_byte_num, num, sizey_bin, sizex_bin)
  simdat = simdat.reshape(num, 1, sizex_sim)
  bindat = bindat.reshape(num, 1, sizey_bin, sizex_bin)

  simdat = torch.tensor(simdat, dtype=torch.float32)
  bindat = torch.tensor(bindat, dtype=torch.float32)
  if (separate_true == True):

    ts, vs = split_and_dataset_ver2(simdat, bindat, train_length, normalize_true, label_img_norm_true)
    return ts, vs

  else:

    if normalize_true == True:
      simdat,_,_ = tensor_norm_DL(simdat)
    if label_img_norm_true == True:
      bindat,_,_ = tensor_norm(bindat) 
    
    dataset = torch.utils.data.TensorDataset(simdat, bindat)
    return dataset




#Make dataset from simdatpath and labelpath(for image identification)
def mysim_to_dataset_ver2(simdat_path, label_path, sizex, num, sim_byte_num, height, width, separate_true, \
  train_length, normalize_true = False):
  """
  Convert one-dimmensional simdat into torch dataset(train and val)
  separate_true:bool (confirm separating dataset into train and val)
  width,height:int (width and height when you make your dataset)
  normalize_true: bool   If this is True, sim-data's average and its std are zero and one.
  label_img_norm_true: bool   If this is True, label-images's value are convert into value from zero to one.
  """
  import py_func.dat_rb_func as drb

  simdat = drb.sim_label_read(simdat_path, sizex, num, False, sim_byte_num)
  label = drb.sim_label_read(label_path, 1, num, True, 4)

  simdat = simdat.reshape(num, 1, height, width)
  label = label.reshape(num,)

  simdat = torch.tensor(simdat, dtype=torch.float32)
  label = label.astype(int)
  label = torch.tensor(label, dtype=torch.int64)

  if (separate_true == True):
    
    ts, vs = split_and_dataset_ver2(simdat, label, train_length, normalize_true, False)
    return ts, vs
    
  else:

    if normalize_true == True:
      simdat,_,_ = tensor_norm_DL(simdat)
    dataset = torch.utils.data.TensorDataset(simdat, label)
    return dataset





#Caluculate mean and std of dataset
def mean_std_dataset(dataset):
  """
  any channel and one-demensional or two-dimensional OK!!
  """
  chans = len(dataset[0][0])
  dat_list = []

  for _ in range(chans):
    dat_channel_list = []
    dat_list.append(dat_channel_list)
  
  for i in range(len(dataset)):
    for j in range(chans):
      #HxW を 1次元のH*Wにしてチャネルのデータにつなげる(*は1次元)
      dat_list[j].append(dataset[i][0][j].reshape(-1))

  mean_list = []
  std_list = []
  for i in range(chans):
    #len(dataset) x tensor(H*W) のリストを tensor(len(dataset)*H*W)に変換(1次元)
    cat_dat = torch.cat(dat_list[i])
    mean_list.append(torch.mean(cat_dat))
    std_list.append(torch.std(cat_dat))
    del cat_dat
  del dat_list

  return mean_list, std_list

#Normalize datasest
def normalize_dataset(dataset, mean_list = None, std_list = None, imaging_cf = False):
    """
    Noramalize dataset
    any channel and one-demensional or two-dimensional OK!!
    both of classification-dataset and imaging-dataset are OK!! 
    meal_list:list of tensor
    std_list:list of tensor
    """
    from torchvision import transforms
    if ((mean_list == None) or (std_list == None)):
        mean_list, std_list = mean_std_dataset(dataset)

    trans = transforms.Normalize(mean = mean_list, std = std_list)
    norm_dat_list = []
    label_list = []
    for i in range(len(dataset)):
        if len(dataset[i][0].shape) == 2:
          #データセットが1次元のときは2次元変換後、標準化し、1次元に戻す

          cha, wid = dataset[i][0].shape
          #1xCx1xW に変換
          tmp = torch.stack([trans(dataset[i][0].reshape(cha, 1, wid))])
          #1xCxW に変換
          tmp = tmp.reshape(1, cha, wid)
        else:
          #このstackは CxHxW を 1xCxHxW に変換
          tmp = torch.stack([trans(dataset[i][0])])
        norm_dat_list.append(tmp)

        if imaging_cf == True:
          tmplabel = torch.stack([dataset[i][1]])
        else:
          tmplabel = dataset[i][1]
        label_list.append(tmplabel)
    
    # len(dataset) x tensor(1xCxHxW) のリストを tensor(len(dataset)xCxHxW)に変換
    cat_dat_norm = torch.cat(norm_dat_list)

    if imaging_cf == True:
      cat_label = torch.cat(label_list)

    del norm_dat_list
    if imaging_cf == True:
      labels = torch.tensor(cat_label,dtype = torch.float32)
    else:
      labels = torch.tensor(label_list,dtype = torch.int64)
    norm_dataset = torch.utils.data.TensorDataset(cat_dat_norm, labels)
    del cat_dat_norm

    return norm_dataset






#Create and save confusion-matrix
def create_cfmat(labels,predictions,matrix_save_path,vmax,figsy,figsx,fontsize):
  import seaborn as sns
  from sklearn.metrics import confusion_matrix

  _, ax = plt.subplots(figsize = (figsy, figsx))
  #plt.rcParams["font.size"] = fontsize

  cm = confusion_matrix(labels,predictions)
  sns.heatmap(cm,square=True,cmap='Blues',annot=True,fmt='d',ax=ax,vmax=vmax,vmin=0,annot_kws={"size":fontsize})
  
  plt.savefig(matrix_save_path)





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








#正則化関数
#L1正則化
def L1norm(model, alpha, loss):
  l1 = torch.tensor(0., requires_grad=True)
  for w in model.parameters():
    #重みの大きさの総和
    l1 = l1 + torch.norm(w, 1)
  loss = loss + alpha*l1
  return loss

#L2正則化
def L2norm(model, lamda, loss):
  l2 = torch.tensor(0., requires_grad=True)
  for w in model.parameters():
    #重みの大きさの2乗和
    l2 = l2 + torch.norm(w)**2
  loss = loss + lamda*l2/2
  return loss







#Deep Learning validation(image classification)
def val_model(dataloader, model, device, lossfunc, predict_label_list_true):
  model.eval()
  correct = 0
  total = 0
  val_loss = 0
  val = []

  if (predict_label_list_true == True):
    predict_list = []
    label_list = []
  
  with torch.no_grad():
    for inputs, labels in dataloader:
      inputs, labels = inputs.to(device), labels.to(device)
      inputs, labels = Variable(inputs), Variable(labels)
      outputs = model(inputs)
      loss = lossfunc(outputs, labels)
      val_loss += loss.item()
      #_, predicted = torch.max(outputs.data, 1)
      predicted = torch.argmax(outputs.data, 1)
      total += labels.size(0)
      correct += (predicted == labels).sum().item()

      if (predict_label_list_true == True):
        for i in range(len(predicted)):
          tmp = int(predicted[i])
          predict_list.append(tmp)
        for i in range(len(labels)):
          tmp = int(labels[i])
          label_list.append(tmp)

    acc = float(correct)/total
    val_loss = val_loss/len(dataloader.dataset)
    val.append(acc)
    val.append(val_loss)

    if (predict_label_list_true == True):
      val.append(predict_list)
      val.append(label_list)
  
  return val

#Deep Learning(image classification)
#MIXUP
def train_model_mixup(dataloader_train, dataloader_val, model, lossfunc, optimizer, epochs, device, L1 = False, alpha = None, L2 = False, lamda = None, mixalpha = 1.0):
    t1=time.time()
    train_loss_list = []
    val_loss_list = []
    train_acc_list = []
    val_acc_list = []

    for epoch in range(epochs):
        model.train()
        for dat_train in dataloader_train:
            inputs, labels = dat_train
            inputs, labels = inputs.to(device), labels.to(device)
            
            #mixup
            inputs, laba, labb, mixlamda = mixup_data(inputs, labels, device, mixalpha)

            #inputs, labels = Variable(inputs), Variable(labels)
            
            inputs, laba, labb = Variable(inputs),  Variable(laba), Variable(labb)
            optimizer.zero_grad()
            outputs = model(inputs)
            #loss = lossfunc(outputs, labels)
            
            loss_mix_func = mixup_criterion(laba, labb, mixlamda)
            loss = loss_mix_func(lossfunc, outputs)

            #正則化(weight decay)
            if L2:
                loss = L2norm(model, lamda, loss)
            elif L1:
                loss = L1norm(model, alpha, loss)


            loss.backward()
            optimizer.step()

        val_val = val_model(dataloader_val, model, device, lossfunc, False)
        val_train = val_model(dataloader_train, model, device, lossfunc, False)
        train_loss_list.append(val_train[1])
        val_loss_list.append(val_val[1])
        train_acc_list.append(val_train[0])
        val_acc_list.append(val_val[0])

        print(f'エポック{epoch+1}------------------------------')
        print(f'val_acc{val_val[0]:.4f} ,train_acc{val_train[0]:.4f}')
        t2=time.time()
        caltime=(t2-t1)/60
        print(f'epochtime:{caltime:.4f}分')
        t1=time.time()

    return train_loss_list, val_loss_list, train_acc_list, val_acc_list




#Deep Learning(image classification)
def train_model_ver3(dataloader_train, dataloader_val, model, lossfunc, optimizer, epochs, device, \
  L1 = False, alpha = None, L2 = False, lamda = None):
  """
  transfer displaying learning_curv from this function.
  select weight decay option
  L1 : bool
  L2 : bool
  alpha and lamda are parameter
  """
  t1=time.time()
  train_loss_list = []
  val_loss_list = []
  train_acc_list = []
  val_acc_list = []
  for epoch in range(epochs):
    model.train()
    for dat_train in dataloader_train:
        inputs, labels = dat_train
        inputs, labels = inputs.to(device), labels.to(device)
        inputs, labels = Variable(inputs), Variable(labels)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = lossfunc(outputs, labels)

        #正則化(weight decay)
        if L2:
          loss = L2norm(model, lamda, loss)
        elif L1:
          loss = L1norm(model, alpha, loss)


        loss.backward()
        optimizer.step()
    
    val_val = val_model(dataloader_val, model, device, lossfunc, False)
    val_train = val_model(dataloader_train, model, device, lossfunc, False)
    train_loss_list.append(val_train[1])
    val_loss_list.append(val_val[1])
    train_acc_list.append(val_train[0])
    val_acc_list.append(val_val[0])

    print(f'エポック{epoch+1}------------------------------')
    print(f'val_acc{val_val[0]:.4f} ,train_acc{val_train[0]:.4f}')
    t2=time.time()
    caltime=(t2-t1)/60
    print(f'epochtime:{caltime:.4f}分')
    t1=time.time()

  return train_loss_list, val_loss_list, train_acc_list, val_acc_list








#Deep Learning validation(decode)
def val_decode_model(dataloader, model, device, lossfunc):
  model.eval()
  val_loss = 0
  val = []
  
  with torch.no_grad():
    for inputs, origin in dataloader:
      inputs, origin = inputs.to(device), origin.to(device)
      inputs, origin = Variable(inputs), Variable(origin)
      outputs = model(inputs)
      loss = lossfunc(outputs, origin)
      val_loss += loss.item()

    
    val_loss = val_loss/len(dataloader.dataset)
    val.append(val_loss)
  
  return val


#Deep Learning(decode)
def train_decode_model_mixup(dataloader_train, dataloader_val, model, lossfunc, optimizer, epochs, device, mixalpha = 1.0):
    t1=time.time()
    train_loss_list=[]
    val_loss_list=[]

    for epoch in range(epochs):
        model.train()
        for dat_train in dataloader_train:
            inputs, origin = dat_train
            inputs, origin = inputs.to(device), origin.to(device)
            # inputs, origin = Variable(inputs), Variable(origin)
            inputs, laba, labb, mixlamda = mixup_data(inputs, origin, device, mixalpha)
            inputs, laba, labb = Variable(inputs), Variable(laba), Variable(labb)

            optimizer.zero_grad()
            outputs = model(inputs)

            # loss = lossfunc(outputs, origin)
            loss_mix_func = mixup_criterion(laba, labb, mixlamda)
            loss = loss_mix_func(lossfunc, outputs)
            loss.backward()
            optimizer.step()

        val_val = val_decode_model(dataloader_val, model, device, lossfunc)
        val_train = val_decode_model(dataloader_train, model, device, lossfunc)
        train_loss_list.append(val_train[0])
        val_loss_list.append(val_val[0])
        t2=time.time()
        caltime=(t2-t1)/60
        print(f'エポック{epoch+1}--------------------------------')
        print(f'epochtime:{caltime}分, train_loss:{val_train[0]}, val_loss:{val_val[0]}')
        t1=time.time()

    return train_loss_list, val_loss_list



#Deep Learning(decode)
def train_decode_model_ver2(dataloader_train, dataloader_val, model, lossfunc, optimizer, epochs, device):
  """
  transfer displaying learning_curv from this function.
  """
  t1=time.time()
  train_loss_list=[]
  val_loss_list=[]

  for epoch in range(epochs):
    model.train()
    for dat_train in dataloader_train:
        inputs, origin = dat_train
        inputs, origin = inputs.to(device), origin.to(device)
        inputs, origin = Variable(inputs), Variable(origin)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = lossfunc(outputs, origin)
        loss.backward()
        optimizer.step()
    val_val = val_decode_model(dataloader_val, model, device, lossfunc)
    val_train = val_decode_model(dataloader_train, model, device, lossfunc)
    train_loss_list.append(val_train[0])
    val_loss_list.append(val_val[0])
    t2=time.time()
    caltime=(t2-t1)/60
    print(f'エポック{epoch+1}--------------------------------')
    print(f'epochtime:{caltime}分, train_loss:{val_train[0]}, val_loss:{val_val[0]}')
    t1=time.time()
  
  return train_loss_list, val_loss_list





#convert tensor into numpy
def tensor_to_numpy(input_tensor, normTrue):
  output_numpy = input_tensor.to('cpu').detach().numpy().copy()
  if normTrue:
    output_numpy *= 255
  output_numpy = np.round(output_numpy)
  output_numpy = np.clip(output_numpy, a_min = 0, a_max = 255)
  output_numpy = output_numpy.astype(np.uint8)
  return output_numpy

#test a decode model and check a output-image
def test_decode_model_and_check_img_ver3(dataloader, img_width, img_height, model, device, figwidth, figheighgt, from_num, to_num, save_dir_path, \
  label_array = None, datanorm = False, correctimgnorm = False):
  """
  In addition to ver2, you can decide number of print-image.
  And calculate PSNR, display label_number.
  from_num>=1 : int
  to_num>=from_num : int
  label_array : you must prepare list or array corresponding to dataloader.
  """
  import numpy as np
  import cv2
  model.eval()
  img_num = 1
  psnrs = []
  with torch.no_grad():
    for inputs, origin in dataloader:
      inputs, origin = inputs.to(device), origin.to(device)
      origin_imgs = origin
      inputs, origin = Variable(inputs), Variable(origin)
      outputs = model(inputs)
      for i in range(len(outputs)):
        output_img_array = tensor_to_numpy(outputs[i], datanorm)
        output_img_array = np.resize(output_img_array,(img_height,img_width))

        origin_img_array = tensor_to_numpy(origin_imgs[i], correctimgnorm)
        origin_img_array = np.resize(origin_img_array,(img_height,img_width))

        psnr = cv2.PSNR(output_img_array, origin_img_array)
        psnrs.append(psnr)

        if from_num <= img_num and img_num <= to_num:
            plt.rcParams["figure.figsize"] = (figwidth, figheighgt)
            fig = plt.figure()

            if label_array is None:
                plt.title('title')
            else :
                plt.title(label_array[img_num - 1])
            plt.subplot(1,2,1)
            plt.imshow(output_img_array)
            plt.gray()
            plt.subplot(1,2,2)
            plt.imshow(origin_img_array)
            plt.gray()
            plt.show()
            fig.savefig(save_dir_path+ '/' + str(i) +'.jpg')

        img_num += 1
  return psnrs



if __name__=='__main__':
    print('Functions related Deep Learning ')

# #test a decode model and check a output-image
# def test_decode_model_and_check_img_ver2(dataloader, img_width, img_height,model, device, figwidth, figheighgt):
#   """
#   In addition to ver1, you can change figure-size.
#   """
#   import numpy as np
#   model.eval()
  
#   with torch.no_grad():
#     for inputs, origin in dataloader:
#       inputs, origin = inputs.to(device), origin.to(device)
#       origin_imgs = origin
#       inputs, origin = Variable(inputs), Variable(origin)
#       outputs = model(inputs)
#       for i in range(len(outputs)):
#         output_img_array = tensor_to_numpy(outputs[i])
#         output_img_array = np.resize(output_img_array,(img_height,img_width))

#         origin_img_array = tensor_to_numpy(origin_imgs[i])
#         origin_img_array = np.resize(origin_img_array,(img_height,img_width))

#         plt.rcParams["figure.figsize"] = (figwidth, figheighgt)
#         plt.subplot(1,2,1)
#         plt.imshow(output_img_array)
#         plt.subplot(1,2,2)
#         plt.gray()
#         plt.imshow(origin_img_array)
#         plt.show()

# #test a decode model and check a output-image
# def test_decode_model_and_check_img(dataloader, img_width, img_height, model, device):
#   import numpy as np
#   model.eval()
  
#   with torch.no_grad():
#     for inputs, origin in dataloader:
#       inputs, origin = inputs.to(device), origin.to(device)
#       origin_imgs = origin
#       inputs, origin = Variable(inputs), Variable(origin)
#       outputs = model(inputs)
#       for i,output_tensor in enumerate(outputs):
#         output_img_array = tensor_to_numpy(output_tensor)

#         output_img_array = np.resize(output_img_array,(img_height,img_width))

#         plt.rcParams["figure.figsize"] = (18, 40)
#         plt.subplot(len(outputs),2,2*i+1)
#         plt.imshow(output_img_array)
      
#       for i,origin_img in enumerate(origin_imgs):
#         origin_img_array = tensor_to_numpy(origin_img)

#         origin_img_array = np.resize(origin_img_array,(img_height,img_width))
        
#         plt.rcParams["figure.figsize"] = (18, 40)
#         plt.subplot(len(outputs),2,2*i+2)
#         plt.imshow(origin_img_array)




# #Deep Learning(image classification)
# def train_model_ver2(dataloader_train, dataloader_val, model, lossfunc, optimizer, epochs, device):
#   """
#   transfer displaying learning_curv from this function.
#   """
#   t1=time.time()
#   train_loss_list = []
#   val_loss_list = []
#   train_acc_list = []
#   val_acc_list = []
#   for epoch in range(epochs):
#     model.train()
#     for dat_train in dataloader_train:
#         inputs, labels = dat_train
#         inputs, labels = inputs.to(device), labels.to(device)
#         inputs, labels = Variable(inputs), Variable(labels)
#         optimizer.zero_grad()
#         outputs = model(inputs)
#         loss = lossfunc(outputs, labels)
#         loss.backward()
#         optimizer.step()
    
#     val_val = val_model(dataloader_val, model, device, lossfunc, False)
#     val_train = val_model(dataloader_train, model, device, lossfunc, False)
#     train_loss_list.append(val_train[1])
#     val_loss_list.append(val_val[1])
#     train_acc_list.append(val_train[0])
#     val_acc_list.append(val_val[0])

#     print(f'エポック{epoch+1}------------------------------')
#     print(f'val_acc{val_val[0]} ,train_acc{val_train[0]}')
#     t2=time.time()
#     caltime=(t2-t1)/60
#     print(f'epochtime:{caltime}分')
#     t1=time.time()

#   return train_loss_list, val_loss_list, train_acc_list, val_acc_list


# #Make dataset from sindatpath and bindatpath(for decoder model)
# def mysimbin_to_dataset(simdat_path, bindat_path, sizex_sim, sizex_bin, sizey_bin, num, sim_byte_num, \
#   bin_byte_num, separate_true, train_length):
#   """
#   Convert one-dimmensional simdat and two-dimmensional simdat into torch dataset(train and val)
#   separate_true:bool (confirm separating dataset into train and val)
#   """
#   import py_func.dat_rb_func as drb
#   simdat = drb.sim_label_read(simdat_path, sizex_sim, num, False, sim_byte_num)
#   bindat = drb.bin_read(bindat_path, bin_byte_num, num, sizey_bin, sizex_bin)
#   simdat = simdat.reshape(num, 1, sizex_sim)
#   bindat = bindat.reshape(num, 1, sizey_bin, sizex_bin)
#   simdat = torch.tensor(simdat, dtype=torch.float32)
#   bindat = torch.tensor(bindat, dtype=torch.float32)
#   if (separate_true == True):
#     # simdat_t,simdat_v = simdat[:train_length],simdat[train_length:]
#     # bindat_t,bindat_v = bindat[:train_length],bindat[train_length:]
#     # dataset_train = torch.utils.data.TensorDataset(simdat_t, bindat_t)
#     # dataset_val = torch.utils.data.TensorDataset(simdat_v, bindat_v)
#     # datasets=[]
#     # datasets.append(dataset_train)
#     # datasets.append(dataset_val)
#     # return datasets
#     ts, vs = split_and_dataset(simdat, bindat, train_length)
#     return ts, vs
#   else:
#     dataset = torch.utils.data.TensorDataset(simdat, bindat)
#     return dataset

# #Make dataset from simdatpath and labelpath(for image identification)
# def mysim_to_dataset(simdat_path,label_path,sizex,num,sim_byte_num,width,height,separate_true,train_length):
#   """
#   Convert one-dimmensional simdat into torch dataset(train and val)
#   separate_true:bool (confirm separating dataset into train and val)
#   width,height:int (width and height when you make your dataset)
#   """
#   import py_func.dat_rb_func as drb

#   simdat = drb.sim_label_read(simdat_path, sizex, num, False, sim_byte_num)
#   label = drb.sim_label_read(label_path, 1, num, True, 4)

#   simdat = simdat.reshape(num,1,width,height)
#   label = label.reshape(num,)

#   simdat = torch.tensor(simdat, dtype=torch.float32)
#   label = label.astype(int)
#   label = torch.tensor(label, dtype=torch.int64)
#   if (separate_true == True):
#     # simdat_t,simdat_v = simdat[:train_length],simdat[train_length:]
#     # label_t,label_v = label[:train_length],label[train_length:]
#     # dataset_train = torch.utils.data.TensorDataset(simdat_t, label_t)
#     # dataset_val = torch.utils.data.TensorDataset(simdat_v, label_v)
#     # datasets=[]
#     # datasets.append(dataset_train)
#     # datasets.append(dataset_val)
#     # return datasets
#     ts, vs = split_and_dataset(simdat, label, train_length)
#     return ts, vs
#   else:
#     dataset = torch.utils.data.TensorDataset(simdat, label)
#     return dataset



# #Normalize datasest
# def normalize_label_dataset(dataset, mean_list = None, std_list = None):
#     """
#     Noramalize dataset
#     label of dataset is one number only(ex: classification-dataset). in case labels are two-dimensional image(ex: imaging-dataset), you cannot use this function.
#     any channel and one-demensional or two-dimensional OK!!
#     meal_list:list of tensor
#     std_list:list of tensor
#     """
#     from torchvision import transforms
#     if ((mean_list == None) or (std_list == None)):
#         mean_list, std_list = mean_std_dataset(dataset)

#     trans = transforms.Normalize(mean = mean_list, std = std_list)
#     norm_dat_list = []
#     label_list = []
#     for i in range(len(dataset)):
#         if len(dataset[i][0].shape) == 2:
#           #データセットが1次元のときは2次元変換後、標準化し、1次元に戻す

#           cha, wid = dataset[i][0].shape
#           #1xCx1xW に変換
#           tmp = torch.stack([trans(dataset[i][0].reshape(cha, 1, wid))])
#           #1xCxW に変換
#           tmp = tmp.reshape(1, cha, wid)
#         else:
#           #このstackは CxHxW を 1xCxHxW に変換
#           tmp = torch.stack([trans(dataset[i][0])])
#         norm_dat_list.append(tmp)
#         tmplabel = dataset[i][1]
#         label_list.append(tmplabel)
    
#     # len(dataset) x tensor(1xCxHxW) のリストを tensor(len(dataset)xCxHxW)に変換
#     cat_dat_norm = torch.cat(norm_dat_list)

#     del norm_dat_list
#     labels = torch.tensor(label_list,dtype = torch.int64)
#     norm_dataset = torch.utils.data.TensorDataset(cat_dat_norm, labels)
#     del cat_dat_norm

#     return norm_dataset




# #Deep Learning(decode)
# def train_decode_model(dataloader_train,dataloader_val,model,lossfunc,optimizer,epochs,device):
#   t1=time.time()
#   train_loss_list=[]
#   val_loss_list=[]

#   for epoch in range(epochs):
#     model.train()
#     for i, dat_train in enumerate(dataloader_train):
#         inputs, origin = dat_train
#         inputs, origin = inputs.to(device), origin.to(device)
#         inputs, origin = Variable(inputs), Variable(origin)
#         optimizer.zero_grad()
#         outputs = model(inputs)
#         loss = lossfunc(outputs, origin)
#         loss.backward()
#         optimizer.step()
#     val_val = []
#     val_train = []
#     val_val = val_decode_model(dataloader=dataloader_val,model=model,device=device,
#                         lossfunc=lossfunc)
#     val_train = val_decode_model(dataloader=dataloader_train,model=model,device=device,
#                           lossfunc=lossfunc)
#     train_loss_list.append(val_train[0])
#     val_loss_list.append(val_val[0])
#     t2=time.time()
#     caltime=(t2-t1)/60
#     print(f'エポック{epoch+1}, epochtime:{caltime}分, train_loss:{val_train[0]}, val_loss:{val_val[0]}')
#     t1=time.time()
  
#   #Display Deep learning curve
#   plt.rcParams["figure.figsize"] = (8,4)
#   plt.subplots_adjust(wspace=0.4, hspace=0.6)
#   plt.plot(range(1, epochs+1), train_loss_list, label="training")
#   plt.plot(range(1, epochs+1), val_loss_list, label="validation")
#   plt.xlabel('Epochs')
#   plt.ylabel('Loss')
#   plt.legend()

#   plt.show()


# #Deep Learning(image classification)
# def train_model(dataloader_train,dataloader_val,model,lossfunc,optimizer,epochs,device):
#   t1=time.time()
#   train_loss_list=[]
#   val_loss_list=[]
#   train_acc_list=[]
#   val_acc_list=[]
#   for epoch in range(epochs):
#     model.train()
#     for i, dat_train in enumerate(dataloader_train):
#         inputs, labels = dat_train
#         inputs, labels = inputs.to(device), labels.to(device)
#         inputs, labels = Variable(inputs), Variable(labels)
#         optimizer.zero_grad()
#         outputs = model(inputs)
#         loss = lossfunc(outputs, labels)
#         loss.backward()
#         optimizer.step()
#     val_val = []
#     val_train = []
#     val_val = val_model(dataloader=dataloader_val,model=model,device=device,lossfunc=lossfunc,predict_label_list_true=False)
#     val_train = val_model(dataloader=dataloader_train,model=model,device=device,lossfunc=lossfunc,predict_label_list_true=False)
#     train_loss_list.append(val_train[1])
#     val_loss_list.append(val_val[1])
#     train_acc_list.append(val_train[0])
#     val_acc_list.append(val_val[0])
#     print(f'エポック{epoch+1} ,val_acc{val_val[0]} ,train_acc{val_train[0]}')
#     t2=time.time()
#     caltime=(t2-t1)/60
#     print(f'epochtime:{caltime}分')
#     t1=time.time()
  
#   plt.rcParams["figure.figsize"] = (8,4)
#   plt.subplots_adjust(wspace=0.4, hspace=0.6)
#   plt.subplot(1,2,1)
#   plt.plot(range(1, epochs+1), train_loss_list, label="training")
#   plt.plot(range(1, epochs+1), val_loss_list, label="validation")
#   plt.xlabel('Epochs')
#   plt.ylabel('Loss')
#   plt.legend()

#   plt.subplot(1,2,2)
#   plt.plot(range(1, epochs+1), train_acc_list, label="training")
#   plt.plot(range(1, epochs+1), val_acc_list, label="validation")
#   plt.xlabel('Epochs')
#   plt.ylabel('Acc')
#   plt.legend()

#   plt.show()


# #convert tensor into numpy
# def tensor_to_numpy(input_tensor):
#   import numpy as np
#   output_numpy = input_tensor.to('cpu').detach().numpy().copy()
#   output_numpy = output_numpy.astype(np.uint8)
#   return output_numpy

# #test a decode model and check a output-image
# def test_decode_model_and_check_img(dataloader,img_width,img_height,model,device):
#   model.eval()
  
#   with torch.no_grad():
#     for inputs, origin in dataloader:
#       inputs, origin = inputs.to(device), origin.to(device)
#       inputs, origin = Variable(inputs), Variable(origin)
#       outputs = model(inputs)
#       for output_tensor in outputs:
#         output_img_array = tensor_to_numpy(output_tensor)
#         plt.imshow(output_img_array)

# #1チャネル,データセット標準化関数
# def one_channel_norm(dataset,mean=None,std=None):
#     """
#     Normalize one-cahnnel dataset
#     """
#     from torchvision import transforms
#     if ((mean==None) and (std==None)):
#         dat_list=[]
#         for i in range(len(dataset)):
#             dat_list.append(dataset[i][0])
#         cat_dat=torch.cat(dat_list)
#         mean=torch.mean(cat_dat)
#         std=torch.std(cat_dat)
#         del cat_dat
        
#     return_list=[]
    
#     trans=transforms.Normalize(mean,std)
#     norm_dat_list=[]
#     label_list=[]
#     for i in range(len(dataset)):
#         tmp=torch.stack([trans(dataset[i][0])])
#         norm_dat_list.append(tmp)
#         tmplabel=dataset[i][1]
#         label_list.append(tmplabel)
    
#     cat_dat_norm=torch.cat(norm_dat_list)
#     del norm_dat_list
#     labels=torch.tensor(label_list,dtype=torch.int64)
#     norm_dataset=torch.utils.data.TensorDataset(cat_dat_norm,labels)
#     del cat_dat_norm
#     return_list.append(norm_dataset)
#     return_list.append(mean)
#     return_list.append(std)

#     return return_list