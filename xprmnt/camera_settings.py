
import cv2
import numpy as np

def decode_fourcc(v):
    v = int(v)
    return "".join([chr((v >> 8 * i) & 0xFF) for i in range(4)])

camera_num = 0

cap = cv2.VideoCapture(camera_num)
#fourcc = decode_fourcc(cap.get(cv2.CAP_PROP_FOURCC))
#fourcc = cap.get(cv2.CAP_PROP_FOURCC)

#width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
#height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
#fps = cap.get(cv2.CAP_PROP_FPS)
#print(f"フォーマット:{fourcc}, 幅:{width}, 高さ:{height}, FPS:{fps}")

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

fourcc = decode_fourcc(cap.get(cv2.CAP_PROP_FOURCC))
width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
fps = cap.get(cv2.CAP_PROP_FPS)
print("fourcc:{} fps:{}　width:{}　height:{}".format(fourcc, fps, width, height))

if cap.isOpened():
    ret, frame = cap.read()
    print(frame.shape)

cap.release()

