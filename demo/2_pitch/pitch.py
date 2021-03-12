#--------------------ViFiDA-----------------------------------------
from vifida import *
os.chdir(os.path.dirname(os.path.realpath(__file__)))   #current folder to this file location
createOutputPath()
#--------------------------------------------------------------------

""" general variables """
framerate = 24
audio = 'do_major'

""" audio processing """
wavfilepath = getWavPath(audio)
audioLoudMono, samplerate, duration = wavToSensed(wavfilepath, framerate, forceMono=True)
audioLoudMono = sensedToFuzzy(audioLoudMono,thres=70)
audioPitch, samplerate, duration = wavToSensed(wavfilepath, framerate,mode='pitch')
audioPitch = sensedToFuzzy(audioPitch)

""" pitch extraction is unstable, we might want to clean some inaccuracies """
for i in range(1,350):
    if audioPitch[i]<0.17:
        audioPitch[i] = (audioPitch[i-1]+audioPitch[i+1])/2
        print(i)    #it is expected to print 3 numbers

""" in/out paths """
infilepath = getMediaPath("moto.jpg")
tempfilepath = getOutputPath("temp.mp4")

""" create a mask for testing """
img = cv2.imread(infilepath)
height, width = img.shape[0:2]
aux = np.linspace(0,height,5,dtype='int')
mask = np.zeros((height, width))
for i in range(height):
    if i in range(aux[1]):
        mask[i,:]=i/aux[1]
    elif i in range(aux[1],aux[3]):
        mask[i,:]=1
    elif i in range(aux[3],aux[4]):
        mask[i,:]=1-(i-aux[3])/(height-1-aux[3])
mask = monocromeToColor(mask)

#1 - Main process with only pitch
pianoO = order(audioPitch, equalizationF)
writeFVideoFromImage(tempfilepath,infilepath,duration,framerate,pianoO)
outfilepath=getOutputPath("1_pitch.mp4")
trackfilepath=getMp3Path(audio)
writeOutputFile(outfilepath,tempfilepath,trackfilepath)

#2 - Main process with pitch modulated by loudness
pianoO = order(audioPitch, equalizationF, modulator=audioLoudMono)
writeFVideoFromImage(tempfilepath,infilepath,duration,framerate,pianoO)
outfilepath=getOutputPath("2_pitchMod.mp4")
trackfilepath=getMp3Path(audio)
writeOutputFile(outfilepath,tempfilepath,trackfilepath)

#3 - Main process with pitch modulated by loudness, using a testing vertical mask
pianoO = order(audioPitch, equalizationF, modulator=audioLoudMono, mask=mask)
writeFVideoFromImage(tempfilepath,infilepath,duration,framerate,pianoO)
outfilepath=getOutputPath("3_pitchModMask.mp4")
trackfilepath=getMp3Path(audio)
writeOutputFile(outfilepath,tempfilepath,trackfilepath)

#expect 5 to 10 minutes of computation when running the whole demo
#on successful end you should get the three video outputs written under ViFiDA\demo\2_pitch\output\
