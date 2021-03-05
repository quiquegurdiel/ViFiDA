#--------------------ViFiDA-----------------------------------------
from vifida import *
import matplotlib.pyplot as plt
import os
from aubio import source, pitch

#Set current folder to project
os.chdir(os.path.dirname(os.path.realpath(__file__)))
createOutputPath()
#--------------------------------------------------------------------

#General variables:
framerate = 24
timewindow = 0.3

#Audio Load
wavfilepath = getMediaPath("do_major.wav")
signal, samplerate, duration = getMonoSignal(wavfilepath)

#Audio processing
""" plt.figure(1)
plt.title("Signal")
plt.plot(signal,color='blue',label='drive')
plt.ylabel("Loudness")
plt.legend()
plt.show() """

odrive = getDriveFromSignal(signal,duration,samplerate,framerate, timewindow = timewindow)
""" autoAnalysis(drive) """
drive = normalizeClippedInterval(odrive,thres=70)

#Define aubio parameters
hop_s = math.ceil(len(signal)/len(drive))
#hop_s = 1
win_s = 4096

#Testing with aubio
s = source(wavfilepath, samplerate, hop_s)
tolerance = 0.8
#Supported methods: `yinfft`, `yin`, `yinfast`, `fcomb`, `mcomb`,
#`schmitt`, `specacf`, `default` (`yinfft`)
pitch_o = pitch("mcomb", win_s, hop_s, samplerate)
pitch_o.set_unit("midi")
pitch_o.set_tolerance(tolerance)

pitches = []
confidences = []

# total number of frames read
total_frames = 0
while True:
    samples, read = s()
    pitch = pitch_o(samples)[0]
    #pitch = int(round(pitch))
    confidence = pitch_o.get_confidence()
    #if confidence < 0.8: pitch = 0.
    #print("%f %f %f" % (total_frames / float(samplerate), pitch, confidence))
    pitches += [pitch]
    confidences += [confidence]
    total_frames += read
    if read < hop_s: break

pitch = normalizeClippedInterval(np.asarray(pitches))
for i in range(1,350):
    if pitch[i]<0.17:
        pitch[i] = (pitch[i-1]+pitch[i+1])/2
        print(i)

""" plt.figure(1)
plt.title("Drives")
plt.plot(drive,color='blue',label='loudness')
plt.plot(pitch,color='green',label='pitch')
plt.plot(drive*drive*pitch,color='red',label='product')
plt.show()

plt.figure(2)
plt.plot(pitches,color='green',label='pitch')
plt.show() """

""" combo = normalizeClippedInterval(pitch*drive,thres=0)
combo2 = normalizeClippedInterval(pitch*drive*drive,thres=0)
combo3 = normalizeClippedInterval(pitch*pitch*drive,thres=0)

plt.figure(2)
plt.plot(combo,color='red',label='pitch*drive')
plt.plot(combo2,color='green',label='pitch*drive*drive')
plt.plot(combo3,color='blue',label='pitch*pitch*drive')
plt.legend()
plt.show() """

#Video processing
videofilepath = getOutputPath("temp.mp4")
imagefilepath = getMediaPath("moto.jpg")
pianoF = [pitch,equalizationF]
writeModulatedFVideoFromImage(drive,videofilepath,imagefilepath,duration,framerate,pianoF)

#Output rendering (video+audio)
outfilepath=getOutputPath("moto.mp4")
trackfilepath=getMediaPath("do_major.mp3")
writeOutputFile(outfilepath,videofilepath,trackfilepath)
