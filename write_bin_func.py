import pandas as pd
import numpy as np
import time
from PIL import Image
import os

if __name__=='__main__':
    print('画素データのバイナリファイル作成関数')

#変更
#10/31 ラベルは4バイト固定、画像データは1バイトor4バイト選択
#10/31 imagenetのラベルファイルを出力


def jpeg_gray_wb_ver2(jpeg_dir_path,bin_dir_path,bin_filename,label_filename,byte_num):
    wnid_list = os.listdir(jpeg_dir_path)
    #classes = int(len(wnid_list))
    image_num_list = [] #各クラスの枚数格納配列
    

    if not os.path.exists(bin_dir_path):
        os.makedirs(bin_dir_path)
    f=open(bin_dir_path+'/'+bin_filename,'wb')
    count=0
    for wnid in wnid_list:
        jpeg_list = os.listdir(jpeg_dir_path+'/'+wnid)
        
        image_num_list.append(int(len(jpeg_list))) #クラス毎に枚数格納
        
        for jpegname in jpeg_list:
            count+=1
            name, ext = os.path.splitext(jpegname)
            if ext == '.JPEG':
                img = Image.open(jpeg_dir_path+'/'+wnid+'/'+jpegname)
                width,height = img.size
                for y in range(height):
                    for x in range(width):
                        tmp = int(img.getpixel((x,y)))
                        if (byte_num==1):
                            tmp=tmp.to_bytes(1,'little')
                        else:
                            tmp=tmp.to_bytes(4,'little')
                        f.write(tmp)  #画像データ書き込み
                img.close()
            if count%100 == 0:
                print(f'{count}枚完了')
    
    label_list=[]
    for i in range(len(image_num_list)):
        for _ in range(image_num_list[i]):
            label_list.append(i)
    
    f2=open(bin_dir_path+'/'+label_filename,'wb')

    for label in label_list:
        label=label.to_bytes(4,'little')
        f2.write(label)
    
    f2.close()
    f.close()

def twodimen_ndarray_wb(data_array,width,height,byte_num,f):
    """
    used for sklearn_wb
    """
    for h in range(height):
        for w in range(width):
            tmp=int(data_array[h][w])
            if (byte_num==1):
                tmp=tmp.to_bytes(1,'little')
            else:
                tmp=tmp.to_bytes(4,'little')
            f.write(tmp)

def sklearn_wb(dataset_number,dirpath,filename,exw,exh,num,byte_num,label_write_true,labelfilename):
    """
    dataset_number=0:mnist
    dataset_number=1:fashion-mnist
    dataset_number=2:cifar10

    if you need full dataset
    num:mnist or fashionmnist :70000
    num:cifar10               :60000
    """
    import cv2
    from sklearn.datasets import fetch_openml
    if (dataset_number==0):
        dataset=fetch_openml('mnist_784')
        size=28
    elif (dataset_number==1):
        dataset=fetch_openml('Fashion-MNIST')
        size=28
    elif (dataset_number==2):
        dataset=fetch_openml('CIFAR_10')
        size=32
        color=3

    x=dataset['data']
    y=dataset['target']
    
    del dataset
    if (type(x)==pd.core.frame.DataFrame):
        x=x.to_numpy()
    if (type(y)==pd.core.frame.DataFrame):
        y=y.to_numpy()
    if (os.path.exists(dirpath) == False):
        os.makedirs(dirpath)
    y=y.astype(int)

    f=open(dirpath+'/'+filename,'wb')
    if (label_write_true==True):
        f2=open(dirpath+'/'+labelfilename,'wb')
    
    t1=time.time()

    for k in range(num):
        #resizeする場合
        if ((exw != size) or (exh != size)):
            if (k==0):
                print("resize")
            if ((dataset_number==0) or (dataset_number==1)):
                original=np.zeros((size,size),np.uint8)
                for h in range(size):
                    for w in range(size):
                        original[h][w]=int(x[k][h*size+w])
                expandimg = cv2.resize(original, dsize=(exw, exh),interpolation=cv2.INTER_CUBIC)
                del original
                
            elif (dataset_number==2):
                original=np.zeros((size,size,color),np.uint8)
                for c in range(color):
                    for h in range(size):
                        for w in range(size):
                            original[h][w][2-c]=int(x[k][(c*size*size)+(h*size+w)])
                expandimg = cv2.resize(original, dsize=(exw, exh),interpolation=cv2.INTER_CUBIC)
                expandimg = cv2.cvtColor(expandimg, cv2.COLOR_BGR2GRAY)
                del original

            twodimen_ndarray_wb(expandimg,exw,exh,byte_num,f)
            del expandimg
        #デフォルトサイズ
        else:
            if (k==0):
                print("default")
            if ((dataset_number==0) or (dataset_number==1)):
                original=np.zeros((size,size),np.uint8)
                for h in range(size):
                    for w in range(size):
                        original[h][w]=int(x[k][h*size+w])
                
            elif (dataset_number==2):
                original=np.zeros((size,size,color),np.uint8)
                for c in range(color):
                    for h in range(size):
                        for w in range(size):
                            original[h][w][2-c]=int(x[k][(c*size*size)+(h*size+w)])
                original = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)

            twodimen_ndarray_wb(original,size,size,byte_num,f)
            del original

        if((k+1)%1000==0):
            print(str(k+1)+'まで終了')
            t2=time.time()
            cal_time=(t2-t1)/60
            print(f"経過時間：{cal_time}分")
        if (label_write_true==True):
            tmplabel=int(y[k])
            tmplabel=tmplabel.to_bytes(4,'little')
            f2.write(tmplabel)            


    del x
    del y
    f.close()
    if (label_write_true==True):
        f2.close()

def twodimen_dataset_wb(dataset,label_write_true,num,width,height,byte_num,dirpath,filename,labelfilename):
  """
  You can use this function to make bindat from pytorch(torchvision) dataset class.
  dataset must be not transformed with tensor
  """
  t1=time.time()
  if (os.path.exists(dirpath) == False):
    os.makedirs(dirpath)
  f=open(dirpath+'/'+filename,'wb')

  if (label_write_true==True):
    f2=open(dirpath+'/'+labelfilename,'wb')

  for i in range(num):
    if (label_write_true==True):
      tmplabel=int(dataset[i][1])
      tmplabel=tmplabel.to_bytes(4,'little')
      f2.write(tmplabel)
    for h in range(height):
      for w in range(width):
        tmp=int(dataset[i][0].getpixel((w,h)))
        if (byte_num==1):
          tmp=tmp.to_bytes(1,'little')
        else:
          tmp=tmp.to_bytes(4,'little')
        f.write(tmp)
    if((i+1)%1000==0):
        print(str(i+1)+'まで終了')
        t2=time.time()
        cal_time=(t2-t1)/60
        print(f"経過時間: {cal_time}分")
  f.close()
  if (label_write_true==True):
    f2.close()


# def sklearn_label_wb(dataset_number,dirpath,filename,num):
#     """
#     dataset_number=0:mnist
#     dataset_number=1:fashion-mnist
#     """
#     from sklearn.datasets import fetch_openml
#     if (dataset_number==0):
#         dataset=fetch_openml('mnist_784')
#     elif (dataset_number==1):
#         dataset=fetch_openml('Fashion-MNIST')
    

#     y=dataset['target']
#     del dataset
#     if (type(y)==pd.core.frame.DataFrame):
#         y=y.to_numpy()
#     y=y.astype(int)
#     if (os.path.exists(dirpath) == False):
#         os.makedirs(dirpath)

#     f=open(dirpath+'/'+filename,'wb')

#     for k in range(num):
#         tmp=int(y[k])
#         tmp=tmp.to_bytes(4,'little')
#         f.write(tmp)
    
#     del y
#     f.close()

# def jpeg_gray_wb(jpeg_dir_path,bin_dir_path,bin_filename):
#     wnid_list = os.listdir(jpeg_dir_path)
    
#     classes = int(len(wnid_list))
    
#     image_num_list = [] #各クラスの枚数格納配列
    
#     if not os.path.exists(bin_dir_path):
#         os.makedirs(bin_dir_path)
#     f=open(bin_dir_path+'/'+bin_filename,'wb')
#     count=0
#     for wnid in wnid_list:
#         jpeg_list = os.listdir(jpeg_dir_path+'/'+wnid)
        
#         image_num_list.append(int(len(jpeg_list))) #クラス毎に枚数格納
        
#         for jpegname in jpeg_list:
#             count+=1
#             name, ext = os.path.splitext(jpegname)
#             if ext == '.JPEG':
#                 img = Image.open(jpeg_dir_path+'/'+wnid+'/'+jpegname)
#                 width,height = img.size
#                 for y in range(height):
#                     for x in range(width):
#                         tmp = int(img.getpixel((x,y)))
#                         tmp=tmp.to_bytes(4,'little')
#                         f.write(tmp)  #画像データ書き込み
#                 img.close()
#             if count%100 == 0:
#                 print(f'{count}枚完了')
    
    
#     classes=classes.to_bytes(4,'little')
#     f.write(classes) #クラス数書き込み
    
#     for num in image_num_list:
#         num=num.to_bytes(4,'little')
#         f.write(num) #画像枚数書き込み
    
#     f.close()
