echo
cd fasterRCNN_FinalProject/tf-faster-rcnn/tools/
mkdir output
cd output/
mkdir res101
wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=1G6tAkrIEmtxsRev-Ed0fTdlVM3slTHqS' -O ~/gdrive-linux-x64
chmod +x ~/gdrive-linux-x64
~/gdrive-linux-x64 download 1DkJEzW3rYScST5gFlS0UkuV3kgKFqqrj --path ~/fasterRCNN_FinalProject/tf-faster-rcnn/tools/output/res101/
cd res101/
tar -xvzf voc_0712_80k-110k.tgz 
cd voc_2007_trainval+voc_2012_trainval/
mkdir default
mv res101_faster_rcnn_iter_110000.* default/
cd ~/fasterRCNN_FinalProject/tf-faster-rcnn/lib/
make