#--------------------ViFiDA-----------------------------------------
from vifida import *
import matplotlib.pyplot as plt
import os
import time
elapsed = []

#Set current folder to project
os.chdir(os.path.dirname(os.path.realpath(__file__)))
createOutputPath()
#--------------------------------------------------------------------

#General variables:
framerate = 24

#Audio Load
wavfilepath = getMediaPath("LightlyRow_100.wav")
signal, samplerate, duration = getMonoSignal(wavfilepath)

#Audio processing
""" plt.figure(1)
plt.title("Signal")
plt.plot(signal,color='blue',label='drive')
plt.ylabel("Loudness")
plt.legend()
plt.show() """

drive = getDriveFromSignal(signal,duration,samplerate,framerate)
""" autoAnalysis(drive) """
drive = normalizeClippedInterval(drive)

#Video processing
videofilepath = getOutputPath("Moto_video.mp4")
imagefilepath = getMediaPath("Moto.jpg")
pianoF = [drive, equalizationF]

def equalizeChannels(img,channels=(0,1,2)):
    for i in channels:
        img[:,:,i] = cv2.equalizeHist(img[:,:,i])
    return img

""" I = cv2.imread(imagefilepath)
imshowQuick(I)
tran = copy.deepcopy(I)
equalizeChannels(tran)
imshowQuick(I)
imshowQuick(tran)
imshowQuick(cv2.addWeighted(I,0.5,tran,0.5,0)) """

start = time.time()
writeFVideoFromImage(videofilepath,imagefilepath,duration,framerate,pianoF)
end = time.time()
elapsed.append(end-start)

#Compute equalization outside loop
def combineF(img,f,tran,gamma=0):
    img = cv2.addWeighted(img,1-f,tran,f,gamma)
    return img

pianoF2=[drive,combineF]

start=time.time()
driveFilters=[pianoF2]
I = cv2.imread(imagefilepath)
height, width = I.shape[0:2]
framecount = math.ceil(duration*framerate)
cvcodec = cv2.VideoWriter_fourcc(*'avc1')
video = cv2.VideoWriter(videofilepath,cvcodec,framerate,(width,height))
tran = copy.deepcopy(I)
equalizeChannels(tran)
for i in range(framecount):
    thisI = I
    for j in range(len(driveFilters)):
        thatDrive = driveFilters[j][0]
        thatFilter = driveFilters[j][1]
        thatF = thatDrive[i]
        thisI = thatFilter(thisI,thatF,tran)
    video.write(thisI)
video.release()
end = time.time()
elapsed.append(end-start)

print(elapsed) #71.73 vs 43.74, notable difference