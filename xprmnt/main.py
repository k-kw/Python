import time
import cv2
import glob
import re
import os
import numpy as np

#import py_func.experiment_func as exp

#画像ディレクトリ
img_dir_path = "kwmt/mnist"

#取得開始画像は何枚目？
start = 60001

#取得画像枚数
img_num = 10000

#画像表示間隔[ms]
sleeptime = 100

#データセットパス
dataset_path = "kwmt/m_60001_70000_fps10_lsd.dat"

camera_num = 0
write_row = 240
miss_size = 640
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
    for i in range(num):
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

#filename_list_sorted = sorted(filename_list)

# for filename in display_file_name_list:
#     print(filename)
print(display_file_name_list[0])
print(len(display_file_name_list))
print(display_file_name_list[len(display_file_name_list) - 1])

#カメラオープン
cap = cv2.VideoCapture(camera_num)
if cap.isOpened():
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
        #frame.shape == 480,640,3
        if ret == False:
            print(f'{i + 1}枚目 : 取得失敗')
            #miss_dataをmiss_size回書き込み
            data_int_wb(miss_data, miss_size, byte)
            
        else:
            #一行書き込み
            frame_line_wb(frame, write_row, f, byte)
            #for data in frame[write_row]:
            #    print(f"{data}, 平均{int(np.mean(data))}")

        cv2.waitKey(int(sleeptime / 2))
        

        # if(i == (img_num - 1)):
        #     break
    
    
    cv2.destroyWindow("view")

    f.close()

else:
    print("camera do not open")

cap.release()



#import time
#import py_func.experiment_func as expf

##画像取得コード
#dataset_path='../dat/experiment/1_11_dat/m_1_5000_exp.dat'
#expf.cam_get_data_ver6(-0.1,5000,1,2,dataset_path,640,0,1,240)

##t_off1=time.time()
###expf.cam_cap_getsize(1)
##dataset_path='../dat/experiment/m_test_num4_akarui_0.8_exp.dat'
##Offset_1 = 0.1
##t_off2=time.time()
##time_off=t_off2-t_off1
##time.sleep(Offset_1 - time_off)
##t = time.time()
##time_off = t - t_off1
##print(f'Offset_1 : {time_off}s')
##expf.cam_get_data_ver3(100,1,1.5,time_off,3,dataset_path,640,0,1,240)
##expf.cam_get_data_ver2(50,1,0.1,dataset_path,640,0,1,240)

##画像取得サイズはcv2.getで確認,cv2.setで変更可





#import py_func.dat_rb_func as drb
#drb.simwave_ver2(4,640,dataset_path,'../dat/label/mnist_label.dat',1,10,10,10,'position','data','./simwave')
##fps = 0.5
##dr = 1
##expf.grayjpgs_to_mp4_ver3('../dat/experiment/mnist_1_5000','./1_5000_dr0.5_fps0.5.mp4',fps,False,dr)
