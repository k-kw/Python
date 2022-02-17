import os
from PIL import Image
import cv2
import py_func.signal_processing_func as sigpf
import numpy as np

def img_write_cv2(fp, img):
  img = img.astype(np.uint8)
  cv2.imwrite(fp, img)

def transform_and_save(img_transform,save_dir_path,img_dir_path):
    dir_list = os.listdir(img_dir_path)
    for wnid_dir in dir_list:
      file_list = os.listdir(img_dir_path+'/'+wnid_dir)
      for file_name in file_list:
        name, ext = os.path.splitext(file_name)
        if ext == '.JPEG':
            img = Image.open(img_dir_path+'/'+wnid_dir+'/'+file_name)
            img = img_transform(img)
            if not os.path.exists(save_dir_path+'/'+wnid_dir):
               os.makedirs(save_dir_path+'/'+wnid_dir)
            img.save(save_dir_path+'/'+wnid_dir+'/'+file_name)
            img.close()

def transform_and_save_rot(img_transform,save_dir_path,img_dir_path,rot_num):
    dir_list = os.listdir(img_dir_path)
    for wnid_dir in dir_list:
      file_list = os.listdir(img_dir_path+'/'+wnid_dir)
      for file_name in file_list:
        name, ext = os.path.splitext(file_name)
        if ext == '.JPEG':
            for i in range(rot_num):
                img = Image.open(img_dir_path+'/'+wnid_dir+'/'+file_name)
                img = img_transform(img)
                if not os.path.exists(save_dir_path+'/'+wnid_dir):
                    os.makedirs(save_dir_path+'/'+wnid_dir)
                img.save(save_dir_path+'/'+wnid_dir+'/'+name+'_'+str(i)+ext)
                img.close()

def hpf_save(save_dir_path,img_dir_path,threshold):
    dir_list = os.listdir(img_dir_path)
    for wnid_dir in dir_list:
      file_list = os.listdir(img_dir_path+'/'+wnid_dir)
      for file_name in file_list:
        name, ext = os.path.splitext(file_name)
        if ext == '.JPEG':
            img = cv2.imread(img_dir_path+'/'+wnid_dir+'/'+file_name,cv2.IMREAD_GRAYSCALE)
            img = sigpf.img_hpf(img,threshold)
            if not os.path.exists(save_dir_path+'/'+wnid_dir):
               os.makedirs(save_dir_path+'/'+wnid_dir)
            img_write_cv2(save_dir_path+'/'+wnid_dir+'/'+file_name,img)

if __name__=='__main__':
    print('Imagenet画像を加工して保存します.')