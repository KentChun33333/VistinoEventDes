
####################################################################################
# Step0
# @import os
# @os.system()

sudo apt-get update
sudo apt-get upgrade
sudo rpi-update

sudo reboot

####################################################################################
# Step1 Dependency Build

sudo apt-get install build-essential cmake pkg-config
sudo apt-get install libjpeg-dev libtiff5-dev libjasper-dev libpng12-dev
sudo apt-get install libavcodec-dev libavformat-dev libswscale-dev libv4l-dev
sudo apt-getinstall libxvidcore-dev libx264-dev
sudo apt-getinstall libgtk2.0-dev
sudo apt-getinstall libatlas-base-dev gfortran
sudo apt-getinstall python2.7-dev python3-dev

####################################################################################
# Step2 Grab the OpenCV source code
# Notice that the version should be the same

cd~
wget-Oopencv.ziphttps://github.com/Itseez/opencv/archive/3.0.0.zip
unzip opencv.zip
wget-Oopencv_contrib.ziphttps://github.com/Itseez/opencv_contrib/archive/3.0.0.zip
unzip opencv_contrib.zip

####################################################################################
# Step3 Setup Python ENV

wget https://bootstrap.pypa.io/get-pip.py
sudo python get-pip.py

sudo pip install virtualenv virtualenvwrapper
sudo rm -rf ~/.cache/pip

# After [virtualenv] and [virtualenvwrapper] have been installed, 
# we need to update [~/.profile] file and insert the folling lines 
# at the bottom of the file

export WORKON_HOME=$HOME/.virtualenvs
source /usr/local/bin/virtualenvwrapper.sh

source ~/.profile
####################################################################################
# Step4 Building the virtualenv ENV.
mkvirtualenv cv
source ~/.profile
workon cv

# must re-build the numpy, it is very important
pip install numpy
cd ~/opencv-3.0.0/
mkdir build
cd build
cmake -D CMAKE_BUILD_TYPE=RELEASE \
-D CMAKE_INSTALL_PREFIX=/usr/local \
-D INSTALL_C_EXAMPLES=ON \
-D INSTALL_PYTHON_EXAMPLES=ON \
-D OPENCV_EXTRA_MODULES_PATH=~/opencv_contrib-3.0.0/modules \
-D BUILD_EXAMPLES=ON ..

#Make sure the python2.7 library


####################################################################################
# Step5 Compile the OpenCv Env.
make -j4

# Note :
# The -j4  switch stands for the number of cores to use when compiling OpenCV. 
# Since we are using a Raspberry Pi 2, we’ll leverage all four cores 
# of the processor for a faster compilation.
# 
# However, if your make  command errors out, 
# I would suggest starting the compilation over again and only using one core
# @make clean
# @make
# 
# Assuming OpenCV compiled without error, 
# all we need to do is install it on our system:

sudo make install
sudo ldconfig

####################################################################################
# Step6 
# Here is to sym-link the OpenCV bindings into the cv virtual environment
#
# Note: In some instances OpenCV can be installed in 
# @/usr/local/lib/python2.7/dist-packages 
# (note the dist-packages  rather than site-packages ).
# If you do not find the cv2.so bindings in site-packages , 
# be sure to check dist-packages  as well.

cd ~/.virtualenvs/cv/lib/python2.7/site-packages/
ln-s/usr/local/lib/python2.7/site-packages/cv2.socv2.so

