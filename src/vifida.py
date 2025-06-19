import numpy as np
import soundfile as sf
#import moviepy.editor as mp
import moviepy as mp
#from cv2 import cv2
import cv2
import math
import sys
import os
import matplotlib.pyplot as plt
from skimage import filters, color, util
#from scikit-image import filters, color, util
from collections import namedtuple
import aubio as au

#--------------------Audio related----------------------------------
def getSignal(filepath, forceMono=False, vervose=1):
    data, samplerate = sf.read(filepath)
    if vervose>1:
        print('----------Audio loaded-------------------')
        print('File path: '+filepath)
        print('Sample rate: '+str(samplerate)+' Hz')

    if forceMono & (len(data.shape) == 2):
        signal = (data[:,0]+data[:,1])/2
        signal = signal[:,]
        if vervose>1:
            print('Channel mode: Forced Mono')
    else:
        signal=data
        if vervose>1:
            if len(data.shape) == 2:
                print('Channel mode: Stereo')
            else:
                print('Channel mode: Mono')

    duration=len(signal)/samplerate
    if vervose>1:
        print('Duration: '+str(round(duration,2))+' s')
        print('-----------------------------------------')

    signal = abs(signal)
    return signal, samplerate, duration

def signalToLoudness(signal, samplerate, timewindow=0.3, ref=10**(-5)):
    loudness = np.zeros_like(signal)
    channels = len(signal.shape)    #1(mono) or 2(stereo)
    samplewindow = math.ceil(timewindow*samplerate)
    if channels==1:
        signal = np.concatenate([np.zeros(samplewindow-1),signal])
        loudness = movingAverage(signal,samplewindow)
    else:   #iterate channels when stereo
        signal = np.concatenate([np.zeros((samplewindow-1,channels)),signal])
        for i in range(len(signal.shape)):
            thisLoud = np.transpose(signal)[i]
            thisLoud = movingAverage(thisLoud,samplewindow)
            loudness[:,i] = thisLoud
    loudness = amplitudeToDecibels(loudness, ref)
    return loudness

def wavToPitch(wavfilepath, signal, samplerate, duration, framerate, win_s=4096, tolerance=0.8, pitchMethod='mcomb'):
    framecount = math.ceil(duration*framerate)
    hop_s = math.ceil(len(signal)/framecount) #aubio parameter, number of audio samples to skip
    s = au.source(wavfilepath, samplerate, hop_s)
    #Supported methods: `yinfft`, `yin`, `yinfast`, `fcomb`, `mcomb`,
    #`schmitt`, `specacf`, `default` (`yinfft`)
    pitch_o = au.pitch(pitchMethod, win_s, hop_s, samplerate)
    pitch_o.set_unit('midi')
    pitch_o.set_tolerance(tolerance)
    pitches = []
    total_frames = 0    #total number of frames read
    while True:
        samples, read = s()
        pitch = pitch_o(samples)[0]
        #pitch = int(round(pitch))
        #if confidence < 0.8: pitch = 0.
        #print("%f %f %f" % (total_frames / float(samplerate), pitch, confidence))
        pitches += [pitch]
        total_frames += read
        if read < hop_s: break
    return np.asarray(pitches)

def wavToSensed(wavfilepath, framerate, mode='loudness', forceMono=False, vervose=1):
    signal, samplerate, duration = getSignal(wavfilepath,forceMono=forceMono, vervose=vervose)
    if mode=='loudness':
        sensed = signalToLoudness(signal,samplerate)
        sensed = quantizeToFrames(sensed,duration,samplerate,framerate)
    elif mode=='pitch':
        sensed = wavToPitch(wavfilepath,signal,samplerate,duration,framerate)
    else:
        sys.exit('unknown mode argument, only "loudness" (default) and "pitch" supported')
    return sensed, samplerate, duration

def quantizeToFrames(sensed, duration, samplerate, framerate):
    framecount = math.ceil(duration*framerate)
    last = math.floor((framecount-1)*samplerate/framerate)
    ind = np.linspace(0,last,framecount)
    ind = ind.astype(int)
    drive = sensed[ind]     #works in stereo too!
    return drive

def sensedToFuzzy(sensed, interval='auto', k=2, thres=30) :
    if interval=='auto':
        m = np.mean(sensed[sensed>thres])
        s = np.std(sensed[sensed>thres])
        interval = (m-k*s,m+k*s)
    fuzzy = (np.clip(sensed,interval[0],interval[1])-interval[0])/(interval[1]-interval[0])
    return fuzzy
#-------------------------------------------------------------------

#--------------------Video related----------------------------------
def getStereoMask(height, width):
    aux = np.linspace(0, 1, width)
    maskR = monocromeToColor(np.array([aux,]*height))
    maskL = 1-maskR
    return [maskL,maskR]

def combineMasks(mask1,mask2):
    norm = 2*mask1*mask2+1-mask1-mask2
    norm[norm==0] = 1 #avoid 0 division
    return mask1*mask2/norm

def applyMask(filtered,original,mask):
    return cv2.add( cv2.multiply(filtered,mask,dtype=8) , cv2.multiply(original,(1-mask),dtype=8) )

def writeFVideoFromVideo(tempfilepath, infilepath, duration, framerate, orders, globalMask=None):
    isGloballyMasked = not globalMask is None
    isAnyStereo = False
    for i in range(len(orders)):
        isAnyStereo = ( isAnyStereo or isStereoOrder(orders[i]) )
    if isAnyStereo:
        stereoOrders=list()
        for i in range(len(orders)):
            stereoOrders.append(forceStereoOrder(orders[i]))
        if isGloballyMasked:
            sys.exit(1)
        else:
            sys.exit(1)
    else:
        if isGloballyMasked:
            sys.exit(1)
        else:
            writeFVideoFromVideoMono(tempfilepath, infilepath, duration, framerate, orders)
    return 0

def writeFVideoFromVideoMono(tempfilepath, infilepath, duration, framerate, orders):
    vidcap = cv2.VideoCapture(infilepath)
    success,img = vidcap.read()
    height, width = img.shape[0:2]
    framecount = math.ceil(duration*framerate)
    cvcodec = cv2.VideoWriter_fourcc(*'avc1')
    video = cv2.VideoWriter(tempfilepath,cvcodec,framerate,(width,height))
    i=0
    while success:
        thisOriginal = img.copy()
        thisFiltered = img.copy()
        for j in range(len(orders)):    #j iterates over orders
            thatOrder = orders[j]
            thatDriver = getattr(thatOrder, 'driver' )
            thatFilter = getattr(thatOrder, 'filter' )
            isMasked = not (getattr(thatOrder, 'mask' ) is None)
            isModulated = not (getattr(thatOrder, 'modulator' ) is None)
            f = thatDriver[i]
            thisFiltered = thatFilter(thisFiltered,f)
            if isMasked & isModulated:
                thatMask = getattr(thatOrder, 'mask' )
                thatModulator = getattr(thatOrder, 'modulator' )
                norm = 2*thatMask*thatModulator[i]+1-thatMask-thatModulator[i]
                norm[norm==0] = 1 #avoid 0 division
                aux = thatMask*thatModulator[i]/norm
                thisFiltered = applyMask(thisFiltered,thisOriginal,aux)
            elif isMasked:
                thatMask = getattr(thatOrder, 'mask')
                thisFiltered = applyMask(thisFiltered,thisOriginal,thatMask)
            elif isModulated:
                thatModulator = getattr(thatOrder, 'modulator')
                thisFiltered = cv2.addWeighted(thisFiltered,thatModulator[i],thisOriginal,1-thatModulator[i],0)
            thisOriginal = thisFiltered.copy()    #next filter acts over the filtered image
        video.write(thisFiltered)
        success,img = vidcap.read()
        i=i+1
    video.release()
    return 0

def writeFVideoFromImage(tempfilepath, infilepath, duration, framerate, orders, globalMask=None):
    isGloballyMasked = not globalMask is None
    isAnyStereo = False
    for i in range(len(orders)):
        isAnyStereo = ( isAnyStereo or isStereoOrder(orders[i]) )
    if isAnyStereo:
        stereoOrders=list()
        for i in range(len(orders)):
            stereoOrders.append(forceStereoOrder(orders[i]))
        if isGloballyMasked:
            writeFVideoFromImageStereoGlobal(tempfilepath, infilepath, duration, framerate, stereoOrders, globalMask)    
        else:
            writeFVideoFromImageStereo(tempfilepath, infilepath, duration, framerate, stereoOrders)
    else:
        if isGloballyMasked:
            writeFVideoFromImageMonoGlobal(tempfilepath, infilepath, duration, framerate, orders, globalMask)
        else:
            writeFVideoFromImageMono(tempfilepath, infilepath, duration, framerate, orders)
    return 0

def writeFVideoFromImageMono(tempfilepath, infilepath, duration, framerate, orders):
    img = cv2.imread(infilepath)
    height, width = img.shape[0:2]
    framecount = math.ceil(duration*framerate)
    cvcodec = cv2.VideoWriter_fourcc(*'avc1')
    video = cv2.VideoWriter(tempfilepath,cvcodec,framerate,(width,height))    
    for i in range(framecount):         #i iterates over frames
        thisOriginal = img.copy()
        thisFiltered = img.copy()
        for j in range(len(orders)):    #j iterates over orders
            thatOrder = orders[j]
            thatDriver = getattr(thatOrder, 'driver' )
            thatFilter = getattr(thatOrder, 'filter' )
            isMasked = not (getattr(thatOrder, 'mask' ) is None)
            isModulated = not (getattr(thatOrder, 'modulator' ) is None)
            f = thatDriver[i]
            thisFiltered = thatFilter(thisFiltered,f)
            if isMasked & isModulated:
                thatMask = getattr(thatOrder, 'mask' )
                thatModulator = getattr(thatOrder, 'modulator' )
                norm = 2*thatMask*thatModulator[i]+1-thatMask-thatModulator[i]
                norm[norm==0] = 1 #avoid 0 division
                aux = thatMask*thatModulator[i]/norm
                thisFiltered = applyMask(thisFiltered,thisOriginal,aux)
            elif isMasked:
                thatMask = getattr(thatOrder, 'mask')
                thisFiltered = applyMask(thisFiltered,thisOriginal,thatMask)
            elif isModulated:
                thatModulator = getattr(thatOrder, 'modulator')
                thisFiltered = cv2.addWeighted(thisFiltered,thatModulator[i],thisOriginal,1-thatModulator[i],0)
            thisOriginal = thisFiltered.copy()    #next filter acts over the filtered image
        video.write(thisFiltered)
    video.release()
    return 0

def writeFVideoFromImageMonoGlobal(tempfilepath, infilepath, duration, framerate, orders, globalMask):
    img = cv2.imread(infilepath)
    height, width = img.shape[0:2]
    framecount = math.ceil(duration*framerate)
    cvcodec = cv2.VideoWriter_fourcc(*'avc1')
    video = cv2.VideoWriter(tempfilepath,cvcodec,framerate,(width,height))    
    for i in range(framecount):         #i iterates over frames
        thisOriginal = img.copy()
        thisFiltered = img.copy()
        thisGlobal = img.copy()
        for j in range(len(orders)):    #j iterates over orders
            thatOrder = orders[j]
            thatDriver = getattr(thatOrder, 'driver' )
            thatFilter = getattr(thatOrder, 'filter' )
            isMasked = not (getattr(thatOrder, 'mask' ) is None)
            isModulated = not (getattr(thatOrder, 'modulator' ) is None)
            f = thatDriver[i]
            thisFiltered = thatFilter(thisFiltered,f)
            if isMasked & isModulated:
                thatMask = getattr(thatOrder, 'mask' )
                thatModulator = getattr(thatOrder, 'modulator' )
                norm = 2*thatMask*thatModulator[i]+1-thatMask-thatModulator[i]
                norm[norm==0] = 1 #avoid 0 division
                aux = thatMask*thatModulator[i]/norm
                thisFiltered = applyMask(thisFiltered,thisOriginal,aux)
            elif isMasked:
                thatMask = getattr(thatOrder, 'mask')
                thisFiltered = applyMask(thisFiltered,thisOriginal,thatMask)
            elif isModulated:
                thatModulator = getattr(thatOrder, 'modulator')
                thisFiltered = cv2.addWeighted(thisFiltered,thatModulator[i],thisOriginal,1-thatModulator[i],0)
            thisOriginal = thisFiltered.copy()    #next filter acts over the filtered image
        video.write( applyMask(thisFiltered,thisGlobal,globalMask) )
    video.release()
    return 0

def writeFVideoFromImageStereo(tempfilepath, infilepath, duration, framerate, orders):
    img = cv2.imread(infilepath)
    height, width = img.shape[0:2]
    stereoMask = getStereoMask(height,width)    #create the stereo mask
    framecount = math.ceil(duration*framerate)
    cvcodec = cv2.VideoWriter_fourcc(*'avc1')
    video = cv2.VideoWriter(tempfilepath,cvcodec,framerate,(width,height))    
    for i in range(framecount):         #i iterates over frames
        thisOriginal = [img.copy(), img.copy()]     #work out each channel sepparately
        thisFiltered = [img.copy(), img.copy()]
        for j in range(len(orders)):    #j iterates over orders
            thatOrder = orders[j]
            thatDriver = getattr(thatOrder, 'driver' )
            thatFilter = getattr(thatOrder, 'filter' )
            isMasked = not (getattr(thatOrder, 'mask' ) is None)
            isModulated = not (getattr(thatOrder, 'modulator' ) is None)
            f = thatDriver[i]
            thisFiltered = [thatFilter(thisFiltered[0],f[0]),thatFilter(thisFiltered[1],f[1])]
            if isMasked & isModulated:
                thatMask = getattr(thatOrder, 'mask' )
                thatModulator = getattr(thatOrder, 'modulator' )
                for k in range(len(thisFiltered)):      #k iterates over channels
                    norm = 2*thatMask*thatModulator[i][k]+1-thatMask-thatModulator[i][k]
                    norm[norm==0] = 1 #avoid 0 division
                    aux = thatMask*thatModulator[i][k]/norm
                    thisFiltered[k] = applyMask(thisFiltered[k],thisOriginal[k],aux)
            elif isMasked:
                thatMask = getattr(thatOrder, 'mask')
                for k in range(len(thisFiltered)):      #k iterates channels
                    thisFiltered[k] = applyMask(thisFiltered[k],thisOriginal[k],thatMask)
            elif isModulated:
                thatModulator = getattr(thatOrder, 'modulator')
                for k in range(len(thisFiltered)):      #k iterates channels
                    thisFiltered[k] = cv2.addWeighted(thisFiltered[k],thatModulator[i][k],thisOriginal[k],1-thatModulator[k][0],0)
            thisOriginal = thisFiltered.copy()    #next filter acts over the filtered image
        video.write(combineStereo(thisFiltered,stereoMask))
    video.release()
    return 0

def writeFVideoFromImageStereoGlobal(tempfilepath, infilepath, duration, framerate, orders, globalMask):
    img = cv2.imread(infilepath)
    height, width = img.shape[0:2]
    stereoMask = getStereoMask(height,width)    #create the stereo mask
    framecount = math.ceil(duration*framerate)
    cvcodec = cv2.VideoWriter_fourcc(*'avc1')
    video = cv2.VideoWriter(tempfilepath,cvcodec,framerate,(width,height))    
    for i in range(framecount):         #i iterates over frames
        thisOriginal = [img.copy(), img.copy()]     #work out each channel sepparately
        thisFiltered = [img.copy(), img.copy()]
        thisGlobal = img.copy()
        for j in range(len(orders)):    #j iterates over orders
            thatOrder = orders[j]
            thatDriver = getattr(thatOrder, 'driver' )
            thatFilter = getattr(thatOrder, 'filter' )
            isMasked = not (getattr(thatOrder, 'mask' ) is None)
            isModulated = not (getattr(thatOrder, 'modulator' ) is None)
            f = thatDriver[i]
            thisFiltered = [thatFilter(thisFiltered[0],f[0]),thatFilter(thisFiltered[1],f[1])]
            if isMasked & isModulated:
                thatMask = getattr(thatOrder, 'mask' )
                thatModulator = getattr(thatOrder, 'modulator' )
                for k in range(len(thisFiltered)):      #k iterates over channels
                    norm = 2*thatMask*thatModulator[i][k]+1-thatMask-thatModulator[i][k]
                    norm[norm==0] = 1 #avoid 0 division
                    aux = thatMask*thatModulator[i][k]/norm
                    thisFiltered[k] = applyMask(thisFiltered[k],thisOriginal[k],aux)
            elif isMasked:
                thatMask = getattr(thatOrder, 'mask')
                for k in range(len(thisFiltered)):      #k iterates channels
                    thisFiltered[k] = applyMask(thisFiltered[k],thisOriginal[k],thatMask)
            elif isModulated:
                thatModulator = getattr(thatOrder, 'modulator')
                for k in range(len(thisFiltered)):      #k iterates channels
                    thisFiltered[k] = cv2.addWeighted(thisFiltered[k],thatModulator[i][k],thisOriginal[k],1-thatModulator[k][0],0)
            thisOriginal = thisFiltered.copy()    #next filter acts over the filtered image
        video.write( applyMask( combineStereo(thisFiltered,stereoMask) ,thisGlobal,globalMask) )
    video.release()
    return 0

def isStereoOrder(subject):
    driver = getattr(subject, 'driver' )
    isStereoDriver = len(driver.shape)==2
    if isStereoDriver:
        return True
    else:
        modulator = getattr(subject, 'modulator' )
        isModulated = not (modulator is None)
        if isModulated:
            isStereoModulator = len(modulator.shape)==2
            return isStereoModulator
        else:
            return False

def forceStereoOrder(old):
    driver = getattr(old, 'driver' )
    isMonoDriver = len(driver.shape)==1
    if isMonoDriver:
        driver = np.transpose([driver,driver])
    fFilter = getattr(old, 'filter' )
    mask = getattr(old, 'mask' )
    modulator = getattr(old, 'modulator' )
    isModulated = not (modulator is None)
    if isModulated:
        isMonoModulated = len(modulator.shape)==1
        if isMonoModulated:
            modulator= np.transpose([modulator,modulator])
    new = order(driver,fFilter,mask,modulator)
    return new

def monocromeToColor(img):
    img = np.transpose(np.array([img,]*3),(1,2,0))
    return img

def combineStereo(stereoImage,stereoMask):
    combined = cv2.multiply(stereoImage[0],stereoMask[0],dtype=8)
    combined = cv2.add(combined, cv2.multiply(stereoImage[1],stereoMask[1],dtype=8) )
    return combined

def writeOutputFile(outfilepath,tempfilepath,audiofilepath,codec='libx264'):
    video = mp.VideoFileClip(tempfilepath)
    video.write_videofile(outfilepath, codec=codec, audio=audiofilepath)
    os.remove(tempfilepath)
    return 0
#-------------------------------------------------------------------

#--------------------Fuzzy filters----------------------------------
#we are using a contrast transform that inputs from -128 to 128, being 0 the identity
#negative numbers reduce thecontrast
#positive value increase it
def contrastF(img, f, amp=140, ori=-40) :
    #map f, in [0,1], to a new interval defined by amp(litude) and ori(gin)
    value = f*amp+ori
    #From https://www.dfstudios.co.uk/articles/programming/image-programming-algorithms/image-processing-algorithms-part-5-contrast-adjustment/
    factor = (259*(value+255))/(255*(259-value))
    aux = img.astype('double')
    aux = factor*(aux-128) + 128
    aux = (aux>=0)*(aux<=255)*aux + (aux>255)*255
    img = aux.astype('uint8')
    return img

#same as contrastF acting only on the value channel
def contrastValueF(img, f, amp=70, ori=0) :
    img = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    #map f, in [0,1], to a new interval defined by amp(litude) and ori(gin)
    value = f*amp+ori
    #From https://www.dfstudios.co.uk/articles/programming/image-programming-algorithms/image-processing-algorithms-part-5-contrast-adjustment/
    factor = (259*(value+255))/(255*(259-value))
    aux = img[:,:,2].astype('double')
    aux = factor*(aux-128) + 128
    aux = (aux>=0)*(aux<=255)*aux + (aux>255)*255
    img[:,:,2] = aux.astype('uint8')
    img = cv2.cvtColor(img,cv2.COLOR_HSV2BGR)
    return img

#redo next filter, the image should be unchanged with f=0
#using this technique you can only desaturate (and it does not make a lot of sense)
def saturationF(img, f, force=[0.7,1]) :
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
        if shape in ('rect','circ'):
            size = powerCorrection(size,1/2)
        size = round(sizemax*size)
    elif size<0:
        size =  size/(-center)
        if shape in ('rect','circ'):
            size = powerCorrection(size,1/2)
        size = round(sizemin*size)
    return int(size)

def getKernelMorfF(shape,size):
    if shape=='rect':
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(size,size))
    elif shape =='circ':
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(size,size))
    elif shape=='cross':
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(size,size))
    return kernel

def applyMorfF(img,f,kernel,center,invert):
    if (f-center)*(-1)**invert>0:
        img = cv2.dilate(img,kernel)
    elif (f-center)*(-1)**invert<0:
        img = cv2.erode(img,kernel)
    return img

def HSVmorfF(img, f, shape='rect', center=0.5, sizemin=50, sizemax=50, invert=0) :
    size = getSizeMorfF(f, shape, center, sizemin, sizemax)
    if size!=0:
        img = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
        kernel = getKernelMorfF(shape,size)
        img = applyMorfF(img,f,kernel,center,invert)
        img = cv2.cvtColor(img,cv2.COLOR_HSV2BGR)
    return img

def RGBmorfF(img, f, shape='rect', center=0.5, sizemin=50, sizemax=50, invert=0) :
    size = getSizeMorfF(f, shape, center, sizemin, sizemax)
    if size!=0:
        kernel = getKernelMorfF(shape,size)
        img = applyMorfF(img,f,kernel,center,invert)
    return img

def noiseF(img, f, top=0.2, mean=0):
    height, width = img.shape[0:2]
    out = np.zeros((height,width*3),dtype='uint8')
    cv2.randn(out,0,top*255*f)
    out = np.resize(out,img.shape)
    out = out+img
    return out

def equalizationF(img, f, top=1, space='RGB'):
    f = f*top #top in (0,1]: the lower the softer the effect
    if space=='HSV':
        tran = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    else:
        tran = img.copy()
    for i in range(3):
        tran[:,:,i] = cv2.equalizeHist(tran[:,:,i])
    if space=='HSV':
        tran = cv2.cvtColor(tran, cv2.COLOR_HSV2BGR)
    return cv2.addWeighted(img,1-f,tran,f,0)

#next is designed to be used with picth and modulated by volume
#curve of colors mechanism still under construction
def colorF(img,f,color=(228,173,44)):
    color = [color[2],color[1],color[0]]    #opencv works in bgr
    #testing some idea________________________
    color[0] = 25 + f*(44-25)
    color[1] = 230 + f*(173-230)
    color[2] = 60 + f*(288-60)
    #_________________________________________
    for i in range(3):
        img[:,:,i] = cv2.addWeighted(img[:,:,i],0.5,color[i],0.5,0,dtype=8)
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
#-------------------------------------------------------------------

#--------------------Orders-----------------------------------------
order = namedtuple('order', ['driver', 'filter', 'mask', 'modulator'])
#next lines define defaults only for mask and modulator, driver and filter are mandatory
order.__new__.__defaults__=(None,None)      #it starts asigning from latest variable
#-------------------------------------------------------------------

#--------------------Maths------------------------------------------
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
#-------------------------------------------------------------------

#--------------------Graphs-----------------------------------------
def autoAnalysis(drive, k=2, thres=30):
    m = np.mean(drive[drive>thres])
    std = np.std(drive[drive>thres])
    plt.figure(1)
    plt.title('Auto mode')
    plt.plot(drive,color='blue',label='drive')
    plt.axhline (m+k*std,color='black', linestyle='dashed')
    plt.axhline (m-k*std,color='black', linestyle='dashed')
    plt.ylabel('Amplitude / Db')
    plt.ylim(0,100)
    plt.legend()
    plt.show()

def getPltColormap(name,n=256):
    cmap = plt.get_cmap(name)
    sm = plt.cm.ScalarMappable(cmap=cmap)
    colormap = sm.to_rgba(np.linspace(0, 1, n))
    return colormap
#-------------------------------------------------------------------

#--------------------Shortcuts--------------------------------------
def getMediaPath(filename=''):
    out = os.getcwd()
    out = os.path.join(out,'media',filename)
    return out

def getWavPath(filename):
    out = getMediaPath(filename+'.wav')
    return out

def getMp3Path(filename):
    out = getMediaPath(filename+'.mp3')
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
#-------------------------------------------------------------------