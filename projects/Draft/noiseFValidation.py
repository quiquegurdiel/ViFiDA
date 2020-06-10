#--------------------ViFiDA-----------------------------------------
from vifida import *
import matplotlib.pyplot as plt
import os
import time

#Set current folder to project
os.chdir(os.path.dirname(os.path.realpath(__file__)))
createOutputPath()
#--------------------------------------------------------------------

#Compare with openCV morfology
elapsed = []

imagefilepath = getMediaPath("Moto.jpg")
I = cv2.imread(imagefilepath)

start = time.time()
J = noiseF(I,1)
end = time.time()
imshowQuick(J)
elapsed.append(end-start)

start = time.time()
J = RGBmorfF(I,1)
end = time.time()
imshowQuick(J)
elapsed.append(end-start)

print(elapsed)
#2.25 vs 0.05, 45 times more than applying a kernel operation with cv2!!!!!

#Separate the inner operations
elapsed = []

img = cv2.imread(imagefilepath)
mode = "gaussian"
mean = 0
top = 0.5
f = 1

start = time.time()
img = util.img_as_float(img)
end = time.time()
elapsed.append(end-start)
if mode in ("gaussian", "speckle"):
    start = time.time()
    img = util.random_noise(img,mode,var=f*top, mean=mean)
    end = time.time()
    elapsed.append(end-start)
if mode in ("salt", "pepper", "s&p"):
    start = time.time()
    img = util.random_noise(img,mode,amount=f*top)
    end = time.time()
    elapsed.append(end-start)
start = time.time()
img = util.img_as_ubyte(img)
end = time.time()
elapsed.append(end-start)

print(elapsed)
#0.03, 0.65, 1.21, time is spent in the type transformation

#Test with an implementation in cv2
elapsed = []
start = time.time()

aux = np.zeros_like(I[:,:,0])
K = np.zeros_like(I)

cv2.randn(aux,0,50)
K[:,:,0] = aux

cv2.randn(aux,0,50)
K[:,:,1] = aux

cv2.randn(aux,0,50)
K[:,:,2] = aux

K = K+I
end = time.time()
elapsed.append(end-start)

imshowQuick(K)

#Implementation 2
start = time.time()
height, width = I.shape[0:2]
K = np.zeros((height,width*3),dtype="uint8")
cv2.randn(K,0,50)
K = np.resize(K,I.shape)
K = K+I
end = time.time()
elapsed.append(end-start)

imshowQuick(K)

#About 0.1, stil longer than morfological operations

#Implementation 3, using GPU
start = time.time()
height, width = I.shape[0:2]
K = cv2.UMat(np.zeros((height,width*3),dtype="uint8"))
cv2.randn(K,0,50)
K = np.resize(K.get(),I.shape)
K = K+I
end = time.time()
elapsed.append(end-start)

print(elapsed)
#1.33, useless.

imshowQuick(K)
