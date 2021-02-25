import numpy as np
import soundfile as sf
import moviepy.editor as mp
from cv2 import cv2
import math
import os
import copy
import matplotlib.pyplot as plt
from skimage import filters, color, util

#Audio related
def getMonoSignal(filepath, vervose=1):
    data, samplerate = sf.read(filepath)
    if vervose>1:
        print("----------Audio loaded-------------------")
        print("File path: "+filepath)
        print("Sample rate: "+str(samplerate)+" Hz")

    if len(data.shape) == 2:
        signal = (data[:,0]+data[:,1])/2
        signal = signal[:,]
        if vervose>1:
            print("Channel mode: Stereo")
    else:
        signal=data
        if vervose>1:
            print("Channel mode: Mono")

    duration=len(signal)/samplerate
    if vervose>1:
        print("Duration: "+str(round(duration,2))+" s")
        print("-----------------------------------------")

    signal = abs(signal)
    return signal, samplerate, duration

def signalToSensed(signal, timewindow, samplerate, ref):
    samplewindow = math.ceil(timewindow*samplerate)
    signal = np.concatenate([np.zeros(samplewindow-1),signal])
    sensed = movingAverage(signal,samplewindow)
    sensed = amplitudeToDecibels(sensed, ref)
    return sensed

def quantizeToFrames(sensed, duration, samplerate, framerate):
    framecount = math.ceil(duration*framerate)
    last = math.floor((framecount-1)*samplerate/framerate)
    ind = np.linspace(0,last,framecount)
    ind = ind.astype(int)
    drive = sensed[ind]
    return drive

def getDriveFromSignal(signal, duration, samplerate, framerate, timewindow = 0.3, ref=10**(-5)):
    sensed = signalToSensed(signal,timewindow,samplerate,ref)
    drive = quantizeToFrames(sensed,duration, samplerate, framerate)
    return drive

def normalizeClippedInterval(vector, interval='auto', k=2, thres=30) :
    if interval=='auto':
        m = np.mean(vector[vector>thres])
        s = np.std(vector[vector>thres])
        interval = (m-k*s,m+k*s)
    vector = (np.clip(vector,interval[0],interval[1])-interval[0])/(interval[1]-interval[0])
    return vector

#filters F
def contrastValueF(img, f, amp=100, ori=-50) :
    img = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)

    #map f, in [0,1], to a new interval defined by amp(litude) and ori(gin)
    value = f*amp+ori

    #From https://www.dfstudios.co.uk/articles/programming/image-programming-algorithms/image-processing-algorithms-part-5-contrast-adjustment/
    factor = (259*(value+255))/(255*(259-value))

    aux = img[:,:,2].astype("double")
    aux = factor*(aux-128) + 128
    aux = (aux>=0)*(aux<=255)*aux + (aux>255)*255
    img[:,:,2] = aux.astype('uint8')

    img = cv2.cvtColor(img,cv2.COLOR_HSV2BGR)

    return img

def contrastF(img, f, amp=140, ori=-40) :
    #map f, in [0,1], to a new interval defined by amp(litude) and ori(gin)
    value = f*amp+ori

    #From https://www.dfstudios.co.uk/articles/programming/image-programming-algorithms/image-processing-algorithms-part-5-contrast-adjustment/
    factor = (259*(value+255))/(255*(259-value))

    aux = img.astype("double")
    aux = factor*(aux-128) + 128
    aux = (aux>=0)*(aux<=255)*aux + (aux>255)*255

    img = aux.astype('uint8')

    return img

def saturationF(img, f, force=[0.8,1.1]) :
    f=f*(force[1]-force[0])+force[0]
    img = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    img[:,:,1] = img[:,:,1]*f
    img = cv2.cvtColor(img,cv2.COLOR_HSV2BGR)
    return img

def rF(img, f) :
    img[:,:,2] = img[:,:,2]*f
    return img

def getSizeMorfF(f, shape, center, sizemin, sizemax):
    size = f-center
    if size>0:
        size =  size/(1-center)
        if shape in ("rect","circ"):
            size = powerCorrection(size,1/2)
        size = round(sizemax*size)
    elif size<0:
        size =  size/(-center)
        if shape in ("rect","circ"):
            size = powerCorrection(size,1/2)
        size = round(sizemin*size)
    return int(size)

def getKernelMorfF(shape,size):
    if shape=="rect":
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(size,size))
    elif shape =="circ":
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(size,size))
    elif shape=="cross":
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(size,size))
    return kernel

def applyMorfF(img,f,kernel,center,invert):
    if (f-center)*(-1)**invert>0:
        img = cv2.dilate(img,kernel)
    elif (f-center)*(-1)**invert<0:
        img = cv2.erode(img,kernel)
    return img

def HSVmorfF(img, f, shape="rect", center=0.5, sizemin=50, sizemax=50, invert=0) :
    size = getSizeMorfF(f, shape, center, sizemin, sizemax)
    if size!=0:
        img = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
        kernel = getKernelMorfF(shape,size)
        img = applyMorfF(img,f,kernel,center,invert)
        img = cv2.cvtColor(img,cv2.COLOR_HSV2BGR)
    return img

def RGBmorfF(img, f, shape="rect", center=0.5, sizemin=50, sizemax=50, invert=0) :
    size = getSizeMorfF(f, shape, center, sizemin, sizemax)
    if size!=0:
        kernel = getKernelMorfF(shape,size)
        img = applyMorfF(img,f,kernel,center,invert)
    return img

""" def noiseF(img, f, mode="gaussian", top=0.05, mean=0):
    img = util.img_as_float(img)
    if mode in ("gaussian", "speckle"):
        img = util.random_noise(img,mode,var=f*top, mean=mean)
    if mode in ("salt", "pepper", "s&p"):
        img = util.random_noise(img,mode,amount=f*top)
    img = util.img_as_ubyte(img)
    return img """

def noiseF(img, f, top=0.2, mean=0):
    height, width = img.shape[0:2]
    out = np.zeros((height,width*3),dtype="uint8")
    cv2.randn(out,0,top*255*f)
    out = np.resize(out,img.shape)
    out = out+img
    return out

def equalizationF(img, f, top=1, space='RGB', gamma=0):
    #tran could be calculated only once if passed as argument (for still images)___________
    if space=='HSV':
        tran = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    else:
        tran = copy.deepcopy(img)

    for i in range(3):
        tran[:,:,i] = cv2.equalizeHist(tran[:,:,i])
    #_____________________________________________________________________________________
    img = cv2.addWeighted(img,1-f,tran,f,gamma)
    return img

def posterizeF(img, f, top=30):
    n = round((f)*(top-1)+1).astype(int)
    indices = np.arange(0,256)
    divider = np.linspace(0,255,n+1)[1]
    quantiz = np.uint8(np.linspace(0,255,n))
    levels = np.clip(np.uint8(indices/divider),0,n-1)
    palette = quantiz[levels]
    img = palette[img]
    return img

#Video related
def writeFVideoFromImage(videofilepath, imagefilepath, duration, framerate, *driveFilters):
    I = cv2.imread(imagefilepath)
    height, width = I.shape[0:2]
    framecount = math.ceil(duration*framerate)
    cvcodec = cv2.VideoWriter_fourcc(*'avc1')
    video = cv2.VideoWriter(videofilepath,cvcodec,framerate,(width,height))
    for i in range(framecount):
        thisI = I
        for j in range(len(driveFilters)):
            thisOldI = thisI
            thatDrive = driveFilters[j][0]
            thatFilter = driveFilters[j][1]
            isMasked = len(driveFilters[j])>2
            thatF = thatDrive[i]
            thisI = thatFilter(thisI,thatF)
            if isMasked:
                thatMask = driveFilters[j][2]
                thisI = cv2.bitwise_and(thisI,thatMask) + cv2.bitwise_and(thisOldI,cv2.bitwise_not(thatMask)) 
        video.write(thisI)
    video.release()
    return 0

def writeOutputFile(outfilepath,videofilepath,trackfilepath,codec='libx264'):
    video = mp.VideoFileClip(videofilepath)
    video.write_videofile(outfilepath, codec=codec, audio=trackfilepath)
    os.remove(videofilepath)
    return 0

#Maths
def movingAverage(vector, n) :
    out = np.cumsum(vector, dtype=float)
    out[n:] = out[n:] - out[:-n]
    out = out[n - 1:] / n
    return out

def amplitudeToDecibels(amplitude, ref):
    return 20*np.log10(amplitude/ref)

def powerCorrection(drive, pow):
    drive = drive**pow
    return drive

#Graphs
def autoAnalysis(drive, k=2, thres=30):
    m = np.mean(drive[drive>thres])
    std = np.std(drive[drive>thres])
    plt.figure(1)
    plt.title("Auto mode")
    plt.plot(drive,color='blue',label='drive')
    plt.axhline (m+k*std,color='black', linestyle='dashed')
    plt.axhline (m-k*std,color='black', linestyle='dashed')
    plt.ylabel("Amplitude / Db")
    plt.ylim(0,100)
    plt.legend()
    plt.show()

def getPltColormap(name,n=256):
    cmap = plt.get_cmap(name)
    sm = plt.cm.ScalarMappable(cmap=cmap)
    colormap = sm.to_rgba(np.linspace(0, 1, n))
    return colormap

#Shorcuts
def getMediaPath(filename=''):
    out = os.getcwd()
    out = os.path.join(out,'media',filename)
    return out

def getOutputPath(filename=''):
    out = os.getcwd()
    out = os.path.join(out,'output',filename)
    return out

def createOutputPath():
    out = os.path.join(os.getcwd(),'output')
    if not os.path.isdir(out):
        os.mkdir(out)
    return 0

def imshowQuick(img,name='1'):
    cv2.namedWindow(name,cv2.WINDOW_NORMAL)
    cv2.imshow(name,img)
    cv2.waitKey(0)
    return 0
