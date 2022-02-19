import numpy as np
from PIL import Image

#変更
#10/31 ラベルは4バイト固定、画像データは1バイトor4バイト選択
#10/31 imagenetのラベル配列出力をwrite_bi_funcへ移行
#10/31 mnistとimagenetのsimdat読み込み統一
#11/8 バイナリデータ読み込み関数を追加
#11/8 配列から画像を作成する関数を追加
#11/8 散乱パターン波形を表示する関数をここに移動

#深層学習前の標準化
def np_normalize_DL(array, mean = None, std = None):
  if (mean == None):
    mean = np.mean(array)
  if (std == None):
    std = np.std(array)
  array = (array - mean)/std
  return array, mean, std

#正規化
def np_normalize(array):
  max = np.max(array)
  min = np.min(array)
  array = (array - min)/(max - min)
  return array


#4byteか1byteずつnumber回読み込みnumpy配列返す
def data_read(data_path,byte,number):
  data_list = []
  with open(data_path,'rb') as f:
    for _ in range(number):
      if (byte == 1):
        tmp = f.read(1)
      elif (byte == 4):
        tmp = f.read(4)
      else:
        print('byte must be 1 or 4')
        return None
      
      data = int.from_bytes(tmp,'little')
      data_list.append(data)

  data_array = np.array(data_list)
  return data_array


#mnistとimagenetとラベル併用
def sim_label_read(data_path, size, num, label_true, byte):
    if label_true == True:
      label = data_read(data_path,4,num)
      label = label.reshape(num,)
      return label
    else:
      data_array = data_read(data_path,byte,num*size)
      data_array = data_array.reshape(num,size)
      return data_array
    # with open(simdat_path,'rb') as f:
    #     if (label_true==True):
    #         for i in range(num):
    #             tmp=f.read(4)
    #             q=int.from_bytes(tmp,'little')
    #             data_list.append(q)
    #     else:
    #         for i in range(num*size):
    #             if (byte_num==1):
    #                 tmp=f.read(1)
    #             else:
    #                 tmp=f.read(4)
    #             q=int.from_bytes(tmp,'little')
    #             data_list.append(q)

    #     data_array=np.array(data_list)
    #     if (label_true==True):
    #         data_array=data_array.reshape(num,)
    #     else:
    #         data_array=data_array.reshape(num,size)

    # return data_array


def simwave_ver2(num,size,simpath,labelpath,byte,dis_width,dis_height,\
  fontsize,xlabel,ylabel,save_dir_path):
  import matplotlib.pyplot as plt

  dat = sim_label_read(simpath,size,num,False,byte)
  label = sim_label_read(labelpath,1,num,True,4)

  plt.rcParams["figure.figsize"] = (dis_width,dis_height)
  plt.rcParams["font.size"] = fontsize
  plt.rcParams["figure.subplot.left"] = 0.15
  
  for i in range(num):
    fig = plt.figure()

    plt.rcParams["figure.figsize"] = (dis_width,dis_height)
    plt.rcParams["font.size"] = fontsize
    plt.rcParams["figure.subplot.left"] = 0.15
    plt.plot(range(0,size), dat[i],linewidth=1)
    

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(label[i])
    fig.savefig(save_dir_path+ '/' + str(i) +'.jpg')

    #plt.show()

def simwave(num,size,simpath,labelpath,byte_num):
  import matplotlib.pyplot as plt

  dat = sim_label_read(simpath,size,num,False,byte_num)
  label = sim_label_read(labelpath,1,num,True,4)
  for i in range(num):
    plt.rcParams["figure.figsize"] = (18, 40)
    plt.subplot(num,2,i+1)
    plt.plot(range(0,size), dat[i],linewidth=1)
    plt.xlabel(label[i])
    plt.ylabel('strength')
  plt.show()

def array_to_grayjpeg(width,height,img_dat,name,save_dir):
  image_buffer=Image.new('L',(width,height))
  for h in range(height):
    for w in range(width):
        pixel=img_dat[h][w]
        image_buffer.putpixel((w,h),(int(pixel),))
  
  image_buffer.save(save_dir+'/'+name+'.jpg','JPEG')
  return 0

def bin_read(bin_path,byte,num,height,width):
  data_array = data_read(bin_path,byte,num*height*width)

  # data_list=[]
  # with open(bin_path,'rb') as f:
  #   for i in range(num*height*width):
  #     if (byte_num==1):
  #       tmp=f.read(1)
  #     else:
  #       tmp=f.read(4)
  #     q=int.from_bytes(tmp,'little')
  #     data_list.append(q)
  # data_array=np.array(data_list)
  data_array = data_array.reshape(num,height,width)
  return data_array



if __name__=='__main__':
    print('Functions related reading binaridata')



        

