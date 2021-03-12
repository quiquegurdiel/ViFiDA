#--------------------ViFiDA-----------------------------------------
from vifida import *
os.chdir(os.path.dirname(os.path.realpath(__file__)))   #current folder to this file location
#--------------------------------------------------------------------

""" load an image just to get its size """
infilepath = getMediaPath("moto.jpg")
I = cv2.imread(infilepath)
height, width = I.shape[0:2]

#1 - create an horizontal mask linearly increasing from 0 to 1
aux = np.linspace(0, 1, width)
maskX = np.array([aux,]*height) #this is the way is done in getStereoMask() by default
imshowQuick(maskX,'1 - horizontal mask')

#2 - create a vertical mask linearly increasing from 0 to 1
aux = np.linspace(1, 0, height)
maskY = np.array([aux,]*width)
maskY = np.transpose(maskY, (1,0))
imshowQuick(maskY,'2 - vertical mask')

#3 - combine both masks vifidawise
combo = combineMasks(maskX,maskY)
imshowQuick(combo,'3 - combination')

#4 - highlight with colors some lines with a constant value of the combination
comboR = combo.copy()
comboG = combo.copy()
comboB = combo.copy()
N = 15      #select a reasonable numbers of values to highlight
tolerance = 0.001
colors = plt.get_cmap('viridis',N)
for i in range(N):
    target = 1/(N+1)*(i+1) 
    color = colors.colors[i]
    aux = combo[(combo>(target-tolerance))&(combo<(target+tolerance))]
    if  not aux.size==0:
        comboR[(combo>(target-tolerance))&(combo<(target+tolerance))] = color[0]
        comboG[(combo>(target-tolerance))&(combo<(target+tolerance))] = color[1]
        comboB[(combo>(target-tolerance))&(combo<(target+tolerance))] = color[2]
combo = np.transpose(np.array([comboR,comboG,comboB]),(1,2,0))
imshowQuick(combo,'4 - combination with lines')

#5 - what would happen without notmalization?
lame = maskX*maskY
lameR = lame.copy()
lameG = lame.copy()
lameB = lame.copy()
colors = plt.get_cmap('viridis',N)
for i in range(N):
    target = 1/(N+1)*(i+1) 
    color = colors.colors[i]
    aux = lame[(lame>(target-tolerance))&(lame<(target+tolerance))]
    if  not aux.size==0:
        lameR[(lame>(target-tolerance))&(lame<(target+tolerance))] = color[0]
        lameG[(lame>(target-tolerance))&(lame<(target+tolerance))] = color[1]
        lameB[(lame>(target-tolerance))&(lame<(target+tolerance))] = color[2]
lame = np.transpose(np.array([lameR,lameG,lameB]),(1,2,0))
imshowQuick(lame,'5 - unnormalized')

""" that is the main point, but you can uncomment below to go trough some divagations """

""" #6 - what if we combine a mask with its negation? (not really sure why would you want to)
test = combineMasks(maskX,1-maskX)
imshowQuick(test,'6 - combine with negative')
#all points have 0.5 value of the combination!
#except in the case (0 and 1), where the combination goes to 0
#is this reasonable? does it have any use at all?
#do we like to have extreme cases? would there be a way to avoid them?

#7 - if we don't normalize we get instead
testLame = maskX * (1-maskX)
imshowQuick(testLame,'7 - unnormalized with negative')
#if we consider the width as 1, we are just intensity-ploting x*(x-1) in every row
#that is a parabola with 0 in both sides and a maximum value of 0.25 in the middle

#so?... nothing really, at least by now, but is quite cool to know it

#p.s.(two days later): if you are ever playing arround and see an unexpected black
#region in the final result (using several masks or masks and modulators) remember this """
