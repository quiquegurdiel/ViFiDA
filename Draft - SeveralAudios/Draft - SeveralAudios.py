from vifida import *
import matplotlib.pyplot as plt
import os

#Set current folder to project
os.chdir(os.path.dirname(os.path.realpath(__file__)))
createOutputPath()

#General variables:
framerate = 24

#Audio Load
wavfilepath = getMediaPath("LightlyRow_100.wav")
signal, samplerate, duration = getMonoSignal(wavfilepath)

#Process audio signal
piano = getDriveFromSignal(signal,duration,samplerate,framerate)
""" autoAnalysis(drive) """
piano = normalizeClippedInterval(piano)

#Mock another drive
cyclic = np.linspace(0,len(piano)-1,len(piano))
cyclic = abs(np.sin(cyclic*30/len(piano)))

#Video processing
videofilepath = getOutputPath("Moto_video.mp4")
imagefilepath = getMediaPath("Moto.jpg")
pianoF = [piano,saturationF]
cyclicF = [cyclic,rF]
writeFVideoFromImage(videofilepath,imagefilepath,duration,framerate,pianoF,cyclicF)

#Output rendering (video+audio)
outfilepath = getOutputPath("Moto.mp4")
trackfilepath = getMediaPath("LightlyRow_100.mp3")
writeOutputFile(outfilepath,videofilepath,trackfilepath)