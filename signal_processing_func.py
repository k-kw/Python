import numpy as np
import matplotlib.pyplot as plt

def img_hpf(img,threshold):
    H=np.zeros(img.shape)
    for h in range(img.shape[0]):
        for w in range(img.shape[1]):
            if (((img.shape[0]/2)-h)**2+((img.shape[1]/2)-w)**2)**0.5>threshold :
                H[h,w]=1
    img=np.fft.fft2(img)
    img=np.fft.fftshift(img)
    img=img*H
    img=np.fft.fftshift(img)
    img=np.fft.ifft2(img)
    img=np.abs(img)
    return img



# def img_read(fname):
#  img = cv2.imread(fname, cv2.IMREAD_GRAYSCALE)
#  return img
# def img_write(fname, img):
#  img = img.astype(np.uint8)
#  cv2.imwrite(fname, img)
#  return 0
# img = img_read("mandrill.bmp")
# img=np.fft.fft2(img)
# img=np.fft.fftshift(img)
# img=np.abs(img)
# plt.imshow(np.log(img.real+1),"gray")
#plt.show()