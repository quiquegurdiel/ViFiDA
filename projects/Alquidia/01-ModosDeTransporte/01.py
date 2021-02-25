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
wavfilepath = getMediaPath("music.wav")
signalMusic, samplerate, duration = getMonoSignal(wavfilepath)

wavfilepath = getMediaPath("voice.wav")
signalVoice, samplerate, duration = getMonoSignal(wavfilepath)

#Audio processing
""" plt.figure(1)
plt.title("Signal")
plt.plot(signal,color='blue',label='drive')
plt.ylabel("Loudness")
plt.legend()
plt.show() """

driveMusic = getDriveFromSignal(signalMusic,duration,samplerate,framerate)
#autoAnalysis(driveMusic,k=1)
driveMusic = normalizeClippedInterval(driveMusic,k=1.5)

driveVoice = getDriveFromSignal(signalVoice,duration,samplerate,framerate)
#autoAnalysis(driveVoice,k=1)
driveVoice = normalizeClippedInterval(driveVoice)

#Video processing
videofilepath = getOutputPath("temp.mp4")
imagefilepath = getMediaPath("img.JPG")

musicF = [driveMusic,saturationF]
voiceF = [driveVoice,contrastValueF]

writeFVideoFromImage(videofilepath,imagefilepath,duration,framerate,voiceF,musicF)

#Output rendering (video+audio)
outfilepath=getOutputPath("main.mp4")
trackfilepath=getMediaPath("master.mp3")
writeOutputFile(outfilepath,videofilepath,trackfilepath)