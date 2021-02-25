#--------------------ViFiDA-----------------------------------------
from vifida import *
import matplotlib.pyplot as plt
import os

#Set current folder to project
os.chdir(os.path.dirname(os.path.realpath(__file__)))
createOutputPath()
#--------------------------------------------------------------------

#General variables:
framerate = 24

#Audio Load
wavfilepath = getMediaPath("Quick.wav")
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
""" pianoF = [drive, saturationF] """
""" def myF (img, f):
    return HSVmorfF(img,f,shape="circ",sizemax=100)
pianoF = [drive, myF] """
""" def myF (img, f):
    return RGBmorfF(img,f,shape="cross",invert=1)
pianoF = [drive, myF] """
""" def myF (img,f):
    return noiseF(img, f, top=0.5)
pianoF = [drive, myF] """
""" pianoF = [drive, equalizationF] """
""" def myF(img,f):
    return equalizationF(img,f,space="HSV")
pianoF = [drive, myF] """
pianoF = [drive,contrastF]
writeFVideoFromImage(videofilepath,imagefilepath,duration,framerate,pianoF)

#Output rendering (video+audio)
outfilepath=getOutputPath("Moto.mp4")
trackfilepath=getMediaPath("Quick.mp3")
writeOutputFile(outfilepath,videofilepath,trackfilepath)
