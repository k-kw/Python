import cv2
import glob
import re
import os
import numpy as np


#画像ディレクトリ
img_dir_path = "kwmt/mnist"

#取得開始画像は何枚目？
start = 1

#取得画像枚数
img_num = 500

#画像表示間隔[ms]
sleeptime = 100

#データセットパス
dataset_path = "kwmt/m_1_5000_fps10_lsd.dat"

getwidth = 1600
getheight = 1200

camera_num = 0
write_row = 600
miss_data = 0
byte = 1

def frame_line_wb(frame, write_row, f, byte):
    """
    byte: int 1 or 4
    """
    for data in frame[write_row]:
        tmp = np.mean(data)
        tmp = int(tmp)
        tmp = tmp.to_bytes(byte, 'little')

        f.write(tmp)


def data_int_wb(data, num, byte):
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
        img = cv2.resize(img, dsize = (4094, 2400))
        cv2.imshow("view", img)
        
        cv2.waitKey(int(sleeptime / 2))

        #画像取得
        ret, frame = cap.read()
        #frame.shape == h,w,3

        if ret == False:
            print(f'{i + 1}枚目 : 取得失敗')
            #miss_dataをmiss_size回書き込み
            data_int_wb(miss_data, getwidth, byte)
            
        else:
            #一行書き込み
            frame_line_wb(frame, write_row, f, byte)
            

        cv2.waitKey(int(sleeptime / 2))
        

        
    
    cv2.destroyWindow("view")

    f.close()

else:
    print("camera do not open")

cap.release()


