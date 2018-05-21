from os import listdir
from os import path
trainvalFile = open('../trainval.txt','w')
testFile = open('../test.txt',"w")
videoList = listdir('./')
videoTrainValList = []
videoTestList = []
id = 0
for i in range(int(len(videoList)*0.8)):
    videoTrainValList += [videoList[i]]
    id = i

for videoName in videoTrainValList:
    print('TrainVal: ',videoName)
    if path.isdir('./'+videoName+'/'):
        frameList = listdir('./'+videoName+'/')
        frameList.sort()
        for frame in frameList:
            trainvalFile.writelines(str(videoName+'/'+frame+'\n').replace(".jpg",""))
trainvalFile.close()

for i in range(i+1,len(videoList)):
    videoTestList += [videoList[i]]
    id = i
for videoName in videoTestList:
    print('Test: ',videoName)
    if path.isdir('./'+videoName+'/'):
        frameList = listdir('./'+videoName+'/')
        frameList.sort()
        for frame in frameList:
            testFile.writelines(str(videoName+'/'+frame+'\n').replace(".jpg",""))
testFile.close()
print('Trainval: ',len(videoTrainValList),'Test: ',len(videoTestList))