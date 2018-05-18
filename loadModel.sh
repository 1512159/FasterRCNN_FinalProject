echo
cd fasterRCNN_FinalProject/tf-faster-rcnn/tools/
mkdir output
cd output/
mkdir res101
wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=1G6tAkrIEmtxsRev-Ed0fTdlVM3slTHqS' -O ~/gdrive-linux-x64
chmod +x ~/gdrive-linux-x64
~/gdrive-linux-x64 download 1DkJEzW3rYScST5gFlS0UkuV3kgKFqqrj --path ~/fasterRCNN_FinalProject/tf-faster-rcnn/tools/output/res101/
cd res101/
cd voc_2007_trainval+voc_2012_trainval/
cd ~/fasterRCNN_FinalProject/tf-faster-rcnn/tools/output/res101
mkdir default
mv voc_0712_80k-110k.tgz default
cd default
tar -xvzf voc_0712_80k-110k.tgz
make