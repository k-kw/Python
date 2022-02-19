import numpy as np
import gc
import matplotlib.pyplot as plt
import cv2

def dataread(data_path, byte, num):
  data_list = []

  with open(data_path,'rb') as f:
    for _ in range(num):
      tmp = f.read(byte)
      data = int.from_bytes(tmp,'little')
      data_list.append(data)

  data_array = np.array(data_list)
  return data_array

class My_numpy:
    #コンストラクタ
    def __init__(self, byte, datapath):
        #byteをnum回読み込み、data_arrayに格納
        self.byte = byte
        self.datapath = datapath
    
    def simread(self, num, sizex):
        self.num = num
        self.sizex = sizex

        self.data = dataread(self.datapath, self.byte, num*sizex)
        self.data = self.dataarray.reshape(num, sizex)


    def binread(self, num, sizey, sizex):
        self.num = num
        self.sizex = sizex
        self.sizey = sizey

        self.data = dataread(self.datapath, self.byte, num*sizex*sizey)
        self.data = self.dataarray.reshape(num, sizey, sizex)


    def labelread(self, num):
        self.data = dataread(self.datapath, 4, num)
    
    
    def save_simwave(self, save_num, labels, dis_width, dis_height, fontsize, save_dir_path):
        plt.rcParams["figure.figsize"] = (dis_width, dis_height)
        plt.rcParams["font.size"] = fontsize
        plt.rcParams["figure.subplot.left"] = 0.15

        for i in range(save_num):
            fig = plt.figure()
            plt.rcParams["figure.figsize"] = (dis_width, dis_height)
            plt.rcParams["font.size"] = fontsize
            plt.rcParams["figure.subplot.left"] = 0.15
            plt.plot(range(0, self.sizex), self.data[i], linewidth=1)
            plt.xlabel("position")
            plt.ylabel("data")
            plt.title(labels[i])
            fig.savefig(save_dir_path+ '/' + str(i) +'.jpg')

    
    def data_to_grayjpg(self, save_num, save_dir_path):
        for i in range(save_num):
            cv2.imwrite(save_dir_path + "/" + str(i) + ".jpg" ,self.data[i])


    #デストラクタ
    def __del__(self):
        del self.data
        gc.collect()




if __name__=='__main__':
    print('Functions related reading binaridata')