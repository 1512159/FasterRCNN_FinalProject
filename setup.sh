export PATH=/opt/bin:/usr/local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/tools/node/bin:/tools/google-cloud-sdk/bin
export LD_LIBRARY_PATH=/usr/lib64-nvidia
export PYTHONPATH=/env/python
echo "======================OKE 1 ======================"
apt-get install software-properties-common
apt-get install python-tk
echo "======================OKE 2 ======================"
apt install vim tmux
apt install fuse
echo "======================OKE 3 ======================"
pip install numpy cython opencv-python easydict==1.6
echo "======================OKE 4 ======================"
apt-get install g++-6 gcc-6 freeglut3-dev build-essential libx11-dev libxmu-dev libxi-dev libglu1-mesa libglu1-mesa-dev g++-6 gcc-6
wget https://developer.nvidia.com/compute/cuda/9.0/Prod/local_installers/cuda_9.0.176_384.81_linux-run
chmod +x cuda_9.0.176_384.81_linux-run
./cuda_9.0.176_384.81_linux-run --override
echo "======================OKE 5 ======================"
ln -s /usr/bin/gcc-6 /usr/local/cuda/bin/gcc
ln -s /usr/bin/g++-6 /usr/local/cuda/bin/g++
echo "======================OKE 6 ======================"