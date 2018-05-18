import shutil
path = 'dummyXML/'
for i in range(1,664):
    shutil.copy2('./example_img00001.xml', path+'img00'+str(i).zfill(3).format(i)+'.xml')
