#--------------------ViFiDA-----------------------------------------
from vifida import *
os.chdir(os.path.dirname(os.path.realpath(__file__)))   #current folder to this file location
createOutputPath()
#--------------------------------------------------------------------

""" general variables """
framerate = 24
audio = 'LightlyRow_100'

""" audio processing """
wavfilepath = getWavPath(audio)
audioLoudMono, samplerate, duration = wavToSensed(wavfilepath, framerate, forceMono=True)
audioLoudMono = sensedToFuzzy(audioLoudMono)

""" in/out paths """
infilepath = getMediaPath('moto.jpg')
tempfilepath = getOutputPath('temp.mp4')

#1- main process example
pianoO = order(audioLoudMono,saturationF)
writeFVideoFromImage(tempfilepath,infilepath,duration,framerate,pianoO)     #ViFiDA pipeline
outfilepath=getOutputPath("1_moto.mp4")
trackfilepath=getMp3Path(audio)
writeOutputFile(outfilepath,tempfilepath,trackfilepath)     #Merge mp4 video with mp3 audio

#on successful end you should get one video output written under ViFiDA\demo\pitch\1_loudness\output\
