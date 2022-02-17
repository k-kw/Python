import time
import cv2
import glob
import re
import os
import numpy as np

def decode_fourcc(v):
    v = int(v)
    return "".join([chr((v >> 8 * i) & 0xFF) for i in range(4)])

def cam_get_format(camera_num):
    """
    カメラの設定を表示
    """
    cap = cv2.VideoCapture(camera_num)
    if cap.isOpened():
        # fourcc = decode_fourcc(cap.get(cv2.CAP_PROP_FOURCC))
        fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
        fourcc = list((fourcc.to_bytes(4, 'little').decode('utf-8')))
        print(fourcc)
        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        fps = cap.get(cv2.CAP_PROP_FPS)
        print("fourcc:{} fps:{} width:{} height:{}".format(fourcc, fps, width, height))
    else:
        print("camera do not open")
    cap.release()



def cam_cap_getsize(camera_num):
    cap = cv2.VideoCapture(camera_num)
    if cap.isOpened():
        ret,frame = cap.read()
        if ret == True:
            print(frame.shape)
    
    cap.release()

def laptime_print(t1,t2,unit,name):
    print_time = t2 - t1
    print(name + ' : ' + str(print_time) + unit)

def cam_get_data_ver6(first_sleep_offset,num,camera_num,sleep_time,dataset_path,miss_size,miss_data,byte_num,write_row):
    """
    get data from a camera
    In additon to ver4, you can set a first_sleep_offset.
    camera doesn't get image until you press a enter key.
    """
    cap=cv2.VideoCapture(camera_num)
    if cap.isOpened():
        f=open(dataset_path,'wb')
        
        #初期オフセット準備
        sleep_offset = first_sleep_offset
        input('press enter and start')
        start_time = time.time()

        for img_number in range(num):
            t1=time.time()
            
            #画像取得
            ret,frame=cap.read()
        
            if ret==False:
                print(f'{img_number+1}枚目 : 取得失敗')
            
                #miss_dataをmiss_size回書き込み
                for _ in range(miss_size):
                    tmp=miss_data
                
                    if (byte_num==1):
                        tmp=tmp.to_bytes(1,'little')
                    else:
                        tmp=tmp.to_bytes(4,'little')
                    f.write(tmp)
                
                continue
            
            else:
                
                #一行書き込み
                for i in range(len(frame[write_row])):
                    tmp=int(frame[write_row][i][0])

                    if (byte_num==1):
                        tmp=tmp.to_bytes(1,'little')
                    else:
                        tmp=tmp.to_bytes(4,'little')
                    f.write(tmp)
            
            t2=time.time()
            act_sleep_time = sleep_time - (t2 - t1)
            time.sleep(act_sleep_time + sleep_offset)
            
            check_time = time.time()
            print(f'{check_time - start_time}経過,{img_number + 2}枚目取得')
            if (check_time - start_time) != float(sleep_time * (img_number + 1)):
                sleep_offset = (sleep_time * (img_number + 1)) - (check_time - start_time)
            else:
                sleep_offset = 0

        f.close()
        
    cap.release()

def cam_get_data_ver5(now_time,first_sleep_offset,num,camera_num,sleep_time,dataset_path,miss_size,miss_data,byte_num,write_row):
    """
    get data from a camera
    In additon to ver4, you can set a first_sleep_offset.
    """
    cap=cv2.VideoCapture(camera_num)
    if cap.isOpened():
        f=open(dataset_path,'wb')
        
        #初期オフセット準備
        sleep_offset = first_sleep_offset

        for img_number in range(num):
            t1=time.time()
            
            #画像取得
            ret,frame=cap.read()
        
            if ret==False:
                print(f'{img_number+1}枚目 : 取得失敗')
            
                #miss_dataをmiss_size回書き込み
                for _ in range(miss_size):
                    tmp=miss_data
                
                    if (byte_num==1):
                        tmp=tmp.to_bytes(1,'little')
                    else:
                        tmp=tmp.to_bytes(4,'little')
                    f.write(tmp)
                
                continue
            
            else:
                
                #一行書き込み
                for i in range(len(frame[write_row])):
                    tmp=int(frame[write_row][i][0])

                    if (byte_num==1):
                        tmp=tmp.to_bytes(1,'little')
                    else:
                        tmp=tmp.to_bytes(4,'little')
                    f.write(tmp)
            
            t2=time.time()
            act_sleep_time = sleep_time - (t2 - t1)
            time.sleep(act_sleep_time + sleep_offset)
            
            check_time = time.time()

            print(check_time - now_time)
            if (check_time - now_time) != float(sleep_time * (img_number + 1)):
                sleep_offset = (sleep_time * (img_number + 1)) - (check_time - now_time)
            else:
                sleep_offset = 0

        f.close()
        
    cap.release()

def cam_get_data_ver4(now_time,num,camera_num,sleep_time,dataset_path,miss_size,miss_data,byte_num,write_row):
    """
    get data from a camera
    calculate Offset of sleeptime from now_time(start time of function) and sleep_time
    """
    cap=cv2.VideoCapture(camera_num)
    if cap.isOpened():
        f=open(dataset_path,'wb')
        
        #初期オフセットである0秒を準備
        t = time.time()
        sleep_offset = time.time() - t

        for img_number in range(num):
            t1=time.time()
            
            #画像取得
            ret,frame=cap.read()
        
            if ret==False:
                print(f'{img_number+1}枚目 : 取得失敗')
            
                #miss_dataをmiss_size回書き込み
                for _ in range(miss_size):
                    tmp=miss_data
                
                    if (byte_num==1):
                        tmp=tmp.to_bytes(1,'little')
                    else:
                        tmp=tmp.to_bytes(4,'little')
                    f.write(tmp)
                
                continue
            
            else:
                
                #一行書き込み
                for i in range(len(frame[write_row])):
                    tmp=int(frame[write_row][i][0])

                    if (byte_num==1):
                        tmp=tmp.to_bytes(1,'little')
                    else:
                        tmp=tmp.to_bytes(4,'little')
                    f.write(tmp)
            
            t2=time.time()
            act_sleep_time = sleep_time - (t2 - t1)
            time.sleep(act_sleep_time + sleep_offset)
            
            check_time = time.time()

            print(check_time - now_time)
            if (check_time - now_time) != float(sleep_time * (img_number + 1)):
                sleep_offset = (sleep_time * (img_number + 1)) - (check_time - now_time)
            else:
                sleep_offset = 0

        f.close()
        
    cap.release()

def cam_get_data_ver3(num,camera_num,k,Offset_1,sleep_time,dataset_path,miss_size,miss_data,byte_num,write_row):
    """
    get data from a camera
    From Offset_1(time before using this function) and Offset_2(time before this function get first image),
     this function appropriately calculate first actual sleep_time
    """
    t_off1=time.time()
    cap=cv2.VideoCapture(camera_num)
    if cap.isOpened():
        f=open(dataset_path,'wb')
        
        t_off2=time.time()
        laptime_print(t_off1,t_off2,'s','Offset_2')

        t_start=time.time()
        
        for img_number in range(num):
            t1=time.time()
            
            #画像取得
            ret,frame=cap.read()
        
            if ret==False:
                print(f'{img_number+1}枚目 : 取得失敗')
            
                #miss_dataをmiss_size回書き込み
                for _ in range(miss_size):
                    tmp=miss_data
                
                    if (byte_num==1):
                        tmp=tmp.to_bytes(1,'little')
                    else:
                        tmp=tmp.to_bytes(4,'little')
                    f.write(tmp)
                
                continue
            
            else:
                
                #一行書き込み
                for i in range(len(frame[write_row])):
                    tmp=int(frame[write_row][i][0])

                    if (byte_num==1):
                        tmp=tmp.to_bytes(1,'little')
                    else:
                        tmp=tmp.to_bytes(4,'little')
                    f.write(tmp)
            
            

            t2=time.time()
            #ほぼ厳密にsleep_time毎に画像取得できるように休止
            if img_number!=0:
                act_sleep_time = sleep_time - (t2 - t1)
                time.sleep(act_sleep_time)
            else:
                #1枚目は オフセット時間 / k を引いて休止
                Offset_2 = t_off2 - t_off1
                act_sleep_time = sleep_time - (t2 - t1) - ((Offset_2 + Offset_1)/k)
                time.sleep(act_sleep_time)

        t_end=time.time()
        laptime_print(t_start,t_end,'s','Alltime')

        f.close()
        
    cap.release()

def cam_get_data_ver2(num,camera_num,sleep_time,dataset_path,miss_size,miss_data,byte_num,write_row):
    """
    get data from a camera
    """
    t_off1=time.time()
    cap=cv2.VideoCapture(camera_num)
    if cap.isOpened():
        f=open(dataset_path,'wb')
        
        #offset2表示
        t_off2=time.time()
        laptime_print(t_off1,t_off2,'s','Offset_2')

        # t_start=time.time()
        
        for img_number in range(num):
            t1=time.time()
            
            #画像取得
            ret,frame=cap.read()
        
            if ret==False:
                print(f'{img_number+1}枚目 : 取得失敗')
            
                #miss_dataをmiss_size回書き込み
                for _ in range(miss_size):
                    tmp=miss_data
                
                    if (byte_num==1):
                        tmp=tmp.to_bytes(1,'little')
                    else:
                        tmp=tmp.to_bytes(4,'little')
                    f.write(tmp)
                
                continue
            
            else:
                
                #一行書き込み
                for i in range(len(frame[write_row])):
                    tmp=int(frame[write_row][i][0])

                    if (byte_num==1):
                        tmp=tmp.to_bytes(1,'little')
                    else:
                        tmp=tmp.to_bytes(4,'little')
                    f.write(tmp)
            
            

            t2=time.time()
            #ほぼ厳密にsleep_time毎に画像取得できるように休止
            time.sleep(sleep_time - (t2 - t1))
            
        # t_end=time.time()
        # laptime_print(t_start,t_end,'s','Alltime')

        f.close()
        
    cap.release()



def grayjpgs_to_mp4_ver3(img_dir_path,movie_path,fps,first_blackimg_true,down_ratio):
    """
    create a movie by appending images in folder
    In ver2, you can add a black image at first.
    You can downsize images and 0-pad at any downsize-ratio(<= 1).
    """
    #画像ディレクトリ内の画像を整列
    img_array = []
    filepath_list=glob.glob(img_dir_path + "/*.jpg")
    filename_list=[]
    for filepath in filepath_list:
        dir_path, filename = os.path.split(filepath)
        filename_list.append(filename)
    
    filename_list_sorted = sorted(filename_list, key=lambda x:int((re.search(r"[0-9]+", x)).group(0)))
    
    #画像をリストに格納
    for filename_sorted in filename_list_sorted:
        img = cv2.imread(img_dir_path + '/' + filename_sorted,cv2.IMREAD_GRAYSCALE)
        img_array.append(img)
    #print(len(img_array))
    height,width = img_array[0].shape
    size = (width,height)
    
    #元画像を縮小
    if (down_ratio != None):
        for i in range(len(img_array)):
            img_tmp = img_array[i]
            pad_width = int((width * (1 / down_ratio) - width) / 2)
            pad_height = int((height * (1 / down_ratio) - height) / 2)
            img_tmp = cv2.copyMakeBorder(img_tmp, pad_width, pad_width, pad_height, pad_height, cv2.BORDER_CONSTANT, 0)
            img_tmp = cv2.resize(img_tmp, size)
            img_array[i] = img_tmp


    #黒画像を先頭に追加
    if first_blackimg_true==True:
        img_black = np.zeros((height ,width), np.uint8)
        img_array = [img_black] + img_array

    out = cv2.VideoWriter(movie_path,cv2.VideoWriter_fourcc(*'MP4V'),fps,size,False)
    for i in range(len(img_array)):
        out.write(img_array[i])

    out.release()

def grayjpgs_to_mp4_ver2(img_dir_path,movie_path,fps,first_blackimg_true):
    """
    create a movie by appending images in folder
    In ver2, you can add a black image at first.
    """
    
    img_array = []
    filepath_list=glob.glob(img_dir_path + "/*.jpg")
    filename_list=[]
    for filepath in filepath_list:
        dir_path, filename = os.path.split(filepath)
        filename_list.append(filename)
    
    filename_list_sorted = sorted(filename_list, key=lambda x:int((re.search(r"[0-9]+", x)).group(0)))
    
    for filename_sorted in filename_list_sorted:
        img = cv2.imread(img_dir_path + '/' + filename_sorted,cv2.IMREAD_GRAYSCALE)
        img_array.append(img)
    #print(len(img_array))
    height,width = img_array[0].shape
    size = (width,height)
    
    #黒画像を先頭に追加
    if first_blackimg_true==True:
        img_black = np.zeros((height ,width), np.uint8)
        img_array = [img_black] + img_array

    out = cv2.VideoWriter(movie_path,cv2.VideoWriter_fourcc(*'MP4V'),fps,size,False)
    for i in range(len(img_array)):
        out.write(img_array[i])

    out.release()

def grayjpgs_to_mp4(img_dir_path,movie_path,fps):
    """
    create a movie by appending images in folder
    """
    img_array = []
    filepath_list=glob.glob(img_dir_path + "/*.jpg")
    filename_list=[]
    for filepath in filepath_list:
        dir_path, filename = os.path.split(filepath)
        filename_list.append(filename)
    
    filename_list_sorted = sorted(filename_list, key=lambda x:int((re.search(r"[0-9]+", x)).group(0)))
    
    for filename_sorted in filename_list_sorted:
        img = cv2.imread(img_dir_path + '/' + filename_sorted,cv2.IMREAD_GRAYSCALE)
        img_array.append(img)

    height,width = img_array[0].shape
    size = (width,height)

    out = cv2.VideoWriter(movie_path,cv2.VideoWriter_fourcc(*'MP4V'),fps,size,False)
    for i in range(len(img_array)):
        out.write(img_array[i])

    out.release()




# def cam_get_data(num,camera_num,sleep_time_first_true,sleep_time_first,sleep_time,dataset_path,miss_size,miss_data,byte_num,write_row):
#     """
#     get data from a camera
#     After sleep_time_first, camera gets a data for every sleep_time
#     """
#     t_off1=time.time()
#     cap=cv2.VideoCapture(camera_num)
#     if cap.isOpened():
#         f=open(dataset_path,'wb')
        
#         t_off2=time.time()
#         laptime_print(t_off1,t_off2,'s','Offset_2')

#         t_start=time.time()
        
#         for img_number in range(num):
#             t1=time.time()
            
#             # #最初の休止
#             # if (sleep_time_first_true==True) and (img_number==0):
#             #     time.sleep(sleep_time_first)

#             #画像取得
#             ret,frame=cap.read()
            
#             # t_get=time.time()
#             # laptime_print(t1,t_get,'s','Gettime')
        
#             if ret==False:
#                 print(f'{img_number+1}枚目 : 取得失敗')
            
#                 #miss_dataをmiss_size回書き込み
#                 for _ in range(miss_size):
#                     tmp=miss_data
                
#                     if (byte_num==1):
#                         tmp=tmp.to_bytes(1,'little')
#                     else:
#                         tmp=tmp.to_bytes(4,'little')
#                     f.write(tmp)
                
#                 continue
            
#             else:
                
#                 #一行書き込み
#                 for i in range(len(frame[write_row])):
#                     tmp=int(frame[write_row][i][0])

#                     if (byte_num==1):
#                         tmp=tmp.to_bytes(1,'little')
#                     else:
#                         tmp=tmp.to_bytes(4,'little')
#                     f.write(tmp)
            
            

#             t2=time.time()
#             #休止
#             time.sleep(sleep_time - (t2 - t1))
#             # t3 = time.time()
#             # laptime_print(t1,t3,'s','Reallaptime')
            
#         t_end=time.time()
#         laptime_print(t_start,t_end,'s','Alltime')

#         f.close()
        
#     cap.release()


# def grayjpgs_to_mp4(img_dir_path,movie_path,fps):
#     img_array = []
#     for filename in sorted(glob.glob(img_dir_path + "/*.jpg")):
#         img = cv2.imread(filename,cv2.IMREAD_GRAYSCALE)
#         img_array.append(img)

#     height,width = img_array[0].shape
#     size = (width,height)

#     out = cv2.VideoWriter(movie_path,cv2.VideoWriter_fourcc(*'MP4V'),fps,size,False)
#     for i in range(len(img_array)):
#         out.write(img_array[i])

#     out.release()