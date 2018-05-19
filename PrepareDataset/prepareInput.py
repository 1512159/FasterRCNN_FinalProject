from os import listdir
from os import path
outputFile = open('trainval.txt','w')
videoList = listdir('./')
for videoName in videoList:
    if path.isdir('./'+videoName+'/'):
        frameList = listdir('./'+videoName+'/')
        for frame in frameList:
            outputFile.writelines(str(videoName+'/'+frame+'\n').replace(".jpg",""))
outputFile.close()