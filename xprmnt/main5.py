#複数列を取得する

import cv2
import glob
import re
import os
from cv2 import COLOR_BGR2GRAY
import numpy as np
import time
t1=time.time()

#画像ディレクトリ
# mnist
img_dir_path = "mnist"
# fashion-mnist
#img_dir_path = "fashionmnist"


#取得開始画像は何枚目？
start = 1

#取得画像枚数
img_num = 1

#画像表示間隔[ms]
sleeptime = 60000

#データセットパス
dataset_path = "expefm_now/none.dat"
#データセット一枚に対して、N＊rowだけデータをとる

getwidth = 1600
getheight = 1200

camera_num = 0

write_row = 600

midrow = 600
#writerow_widthは偶数
writerow_width = 4


miss_data = 0
byte = 1

#N回だけ同じ画像からデータ取得
N = 3


screenwidth = 4096
screenheight = 2400

#真ん中の複数列を取得
def frame_multiline_wb(frame, midrow, writerow_width, f, byte):
    """
    byte: int 1 or 4
    """
    #グレースケール化
    frame = cv2.cvtColor(frame, COLOR_BGR2GRAY)

    startrow_num = int(midrow - (writerow_width/2))
    write_rows = frame[range(startrow_num, startrow_num + writerow_width), :]

    
    for writedata in write_rows:
        for data in writedata:
            data = int(data)
            data = data.to_bytes(byte, 'little')

            f.write(data)


#真ん中複数列の平均を取得
def frame_multi_mean_wb(frame, midrow, writerow_width, f, byte):
    #グレースケール化
    frame = cv2.cvtColor(frame, COLOR_BGR2GRAY)

    #書き込む行を取り出す
    startrow_num = int(midrow - (writerow_width/2))
    write_rows = frame[range(startrow_num, startrow_num + writerow_width), :]

    
    #列平均
    write_rows = np.mean(write_rows, axis = 0)
    
    for data in write_rows:
        data = int(data)
        data = data.to_bytes(byte, 'little')

        f.write(data)


def frame_line_wb(frame, write_row, f, byte):
    """
    byte: int 1 or 4
    """
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    for data in frame[write_row]:
        data = int(data)
        data = data.to_bytes(byte, 'little')

        f.write(data)


def data_int_wb(data, num, f, byte):
    """
    byte: int 1 or 4
    """
    for _ in range(num):
        tmp = int(data)
        tmp = tmp.to_bytes(byte, 'little')

        f.write(tmp)

#画像ディレクトリ内の画像を整列
img_array = []
filepath_list = glob.glob(img_dir_path + "/*.jpg")
filename_list = []
for filepath in filepath_list:
    _, filename = os.path.split(filepath)
    filename_list.append(filename)
filename_list_sorted = sorted(filename_list, key=lambda x:int((re.search(r"[0-9]+", x)).group(0)))

#表示させる画像を選ぶ,何枚目から,何枚
starttmp = start - 1
display_file_name_list = filename_list_sorted[starttmp : starttmp + img_num]


print(display_file_name_list[0])
print(len(display_file_name_list))
print(display_file_name_list[len(display_file_name_list) - 1])

#カメラオープン
cap = cv2.VideoCapture(camera_num)
if cap.isOpened():

    #取得サイズ設定
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, getwidth)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, getheight)


    f = open(dataset_path,'wb')
    

    cv2.namedWindow("view", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("view", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)


    for i, filename_sorted in enumerate(display_file_name_list):
        #画像表示
        
        img = cv2.imread(img_dir_path + '/' + filename_sorted, cv2.IMREAD_GRAYSCALE)

        #resize(dsize = (width, height))
        img = cv2.resize(img, dsize = (screenwidth, screenheight))
        cv2.imshow("view", img)
        
        cv2.waitKey(int(sleeptime))

        #同じ画像に対してN枚画像取得枚
        for i in range(N):
            ret, frame = cap.read()
            #print(frame.shape)
            #frame.shape == h,w,3
            if ret == False:
                print(f'{i + 1}枚目 : 取得失敗')
                #miss_dataをmiss_size回書き込み
                data_int_wb(miss_data, getwidth, f, byte)

                # #miss_dataをget_with*writerow_width回書き込み
                # data_int_wb(miss_data, getwidth * writerow_width, f, byte)

            else:
                #一行書き込み
                frame_line_wb(frame, write_row, f, byte)
                
                # #複数行書き込み
                # frame_multiline_wb(frame, midrow, writerow_width, f, byte)

                # #複数行平均して書き込み
                # frame_multi_mean_wb(frame, midrow, writerow_width, f, byte)


        #cv2.waitKey(int(sleeptime / 2))
        

        
    
    cv2.destroyWindow("view")

    f.close()

else:
    print("camera do not open")

cap.release()

t2=time.time()

print((t2-t1)/60)