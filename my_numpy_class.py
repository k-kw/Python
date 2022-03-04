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
    
    def ndarraytodata(self, input_ndarray):
        self.data = input_ndarray
    
    def simread(self, num, sizex):
        self.num = num
        self.sizex = sizex

        self.data = dataread(self.datapath, self.byte, num*sizex)
        self.data = self.data.reshape(num, sizex)


    def binread(self, num, sizey, sizex):
        self.num = num
        self.sizex = sizex
        self.sizey = sizey

        self.data = dataread(self.datapath, self.byte, num*sizex*sizey)
        self.data = self.data.reshape(num, sizey, sizex)


    def labelread(self, num):
        self.data = dataread(self.datapath, 4, num)
    
    
    def save_simwave_old(self, save_num, labels, dis_width, dis_height, fontsize, save_dir_path):
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

    def save_simwave(self, save_start_num, save_num, labels, dis_width, dis_height, fontsize, save_dir_path):
        plt.rcParams["figure.figsize"] = (dis_width, dis_height)
        plt.rcParams["font.size"] = fontsize
        plt.rcParams["figure.subplot.left"] = 0.15

        for i in range(save_start_num - 1, save_start_num - 1 + save_num):
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
    
    def normalize(self):
        self.max = np.max(self.data)
        self.min = np.min(self.data)
        self.data = (self.data - self.min) / (self.max - self.min)
    
    def normalize_DL(self, mean = None, std = None):
        if mean == None:
            mean = np.mean(self.data)
        if std == None:
            std = np.std(self.data)
        self.data = (self.data - mean) / std


    def writebin(self, data):
        """
        selfとは別の配列を、__init__で用意したパスに書き込む
        """
        data = data.reshape(-1)

        with open(self.datapath,'wb') as f:
            for dat in data:
                dat = int(dat)
                dat = dat.to_bytes(self.byte,'little')
                f.write(dat)

        
        
    #デストラクタ
    def __del__(self):
        del self.data
        gc.collect()





if __name__=='__main__':
    print('Functions related reading binaridata')