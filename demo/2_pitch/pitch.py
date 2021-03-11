#--------------------ViFiDA-----------------------------------------
from vifida import *
import matplotlib.pyplot as plt
import os
from aubio import source, pitch

#Set current folder to project
os.chdir(os.path.dirname(os.path.realpath(__file__)))
createOutputPath()
#--------------------------------------------------------------------

""" general variables """
framerate = 24
timewindow = 0.3

""" audio processing """
wavfilepath = getMediaPath("do_major.wav")
signal, samplerate, duration = getMonoSignal(wavfilepath)
drive = getDriveFromSignal(signal,duration,samplerate,framerate, timewindow = timewindow)
#autoAnalysis(drive)
driveF = normalizeClippedInterval(drive,thres=70)

""" pitch extraction using aubio and driver construction """
hop_s = math.ceil(len(signal)/len(driveF)) #aubio parameter, number of audio samples to skip
win_s = 4096        #aubio parameter, window of fourier transform
s = source(wavfilepath, samplerate, hop_s)
tolerance = 0.8
#Supported methods: `yinfft`, `yin`, `yinfast`, `fcomb`, `mcomb`,
#`schmitt`, `specacf`, `default` (`yinfft`)
pitch_o = pitch("mcomb", win_s, hop_s, samplerate)      #'mcomb' works the best here
pitch_o.set_unit("midi")
pitch_o.set_tolerance(tolerance)

pitches = []
confidences = []
total_frames = 0    #total number of frames read
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

""" pitch extraction is unstable, we might want to clean some inaccuracies """
for i in range(1,350):
    if pitch[i]<0.17:
        pitch[i] = (pitch[i-1]+pitch[i+1])/2
        #print(i) #see the problematic points

""" in/out paths and mask """
infilepath = getMediaPath("moto.jpg")
tempfilepath = getOutputPath("temp.mp4")
mask = createStereoMask(infilepath)

#1 - Main process with only pitch
pianoO = order(pitch, equalizationF)
writeFVideoFromImage_beta(tempfilepath,infilepath,duration,framerate,pianoO)
outfilepath=getOutputPath("1_pitchonly.mp4")
trackfilepath=getMediaPath("do_major.mp3")
writeOutputFile(outfilepath,tempfilepath,trackfilepath)

#2 - Main process with pitch modulated by loudness
pianoO = order(pitch, equalizationF, modulator=driveF)
writeFVideoFromImage_beta(tempfilepath,infilepath,duration,framerate,pianoO)
outfilepath=getOutputPath("2_pitchmodulated.mp4")
trackfilepath=getMediaPath("do_major.mp3")
writeOutputFile(outfilepath,tempfilepath,trackfilepath)

#3 - Main process with pitch modulated by loudness, using left channel mask
pianoO = order(pitch, equalizationF, modulator=driveF, mask=mask[0])
writeFVideoFromImage_beta(tempfilepath,infilepath,duration,framerate,pianoO)
outfilepath=getOutputPath("3_pitchmodulatedL.mp4")
trackfilepath=getMediaPath("do_major.mp3")
writeOutputFile(outfilepath,tempfilepath,trackfilepath)

#expect 5 to 10 minutes of computation when running the whole demo
#on successful end you should get the three video outputs written under ViFiDA\demo\pitch\output