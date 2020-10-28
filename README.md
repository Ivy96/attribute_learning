# attribute_learning
Learns a model for object attribute classfication

git clone https://github.com/baniks/attribute_learning.git

cd attribute_learning

# Dependencies:
### create virtualenv
```
virtualenv -p /usr/bin/python2.7 venv_attrib
source venv_attrib/bin/activate

pip install numpy  
pip install Pillow

pip install scipy

pip install scikit-image

pip install sklearn

pip install lmdb

pip install opencv-python==3.4.0.14
```
### install caffe from [here](https://github.com/baniks/caffe-master.git)
New layers: loss_weight_layer, multi_label_accuracy_layer, weighted_sigmoid_cross_entropy_loss_layer

Follow the instruction from http://caffe.berkeleyvision.org/installation.html. 

In short:

git clone https://github.com/baniks/caffe-master.git

cd caffe-master

cp Makefile.config.example Makefile.config

* Put the virtualenv python path in the Makefile.config file.

PYTHON_INCLUDE := <PATH_TO_venv_attrib>/include/python2.7 \
    <PATH_TO_venv_attrib>/lib/python2.7/site-packages/numpy/core/include

* Add path to hdf5 in INCLUDE_DIRS and LIBRARY_DIRS path  
```
INCLUDE_DIRS := $(PYTHON_INCLUDE) /usr/local/include /usr/include/hdf5/serial/<br/>
LIBRARY_DIRS := $(PYTHON_LIB) /usr/local/lib /usr/lib /usr/lib/x86_64-linux-gnu/hdf5/serial/ 
```
* Opencv requirement
```
OPENCV_VERSION := 3
```
* Make other necessary changes in Makefle.config for CPU/GPU use, and cuda.
```
make clean<br/>
make -j8 all
```
### Install pycaffe 
`cd <caffe-home/python>`
`for req in $(cat requirements.txt); do pip install $req; done`
`cd <caffe-home>`
`make pycaffe`

Another guide for caffe installation : http://installing-caffe-the-right-way.wikidot.com/start

# Dataset:
* Download Imagenet dataset from [here](http://image-net.org/index) for the [imagenet-attribute synsets](http://image-net.org/api/text/imagenet.attributes.obtain_synset_wordlist). Extract the images in a directory named "imagenet_attrib/orig".
* Download the attribute annotation file [attrann.mat](http://image-net.org/download-attributes).
* Use the split_annotation function from project_root/code/prepare_data.py to split it into 4 smaller annotation files.

# Prepare dataset:
* Preprocessing image files
`python prepare_data.py --data_root <PATH_TO_imagenet_attrib/imagenet_attrib/> --task 1`

train/val/test list files will be created in <project_root>/data/

preprocessed image will be saved in <PATH_TO_imagenet_attrib/imagenet_attrib/cropped_to_bbox>

* Create data lmdb file
```
./create_data_lmdb.sh DS=ia LMDBFILE_PATH=<project_root>/data ANNFILE_PATH=<project_root>/data CAFFE_TOOL_PATH=<caffe_root>/build/tools DATA_ROOT=<PATH_TO_imagenet_attrib/imagenet_attrib/cropped_to_bbox/>
```
LMDB files will be created in <project_root>/data/


# Train model Deep Attribute Network (DAN)
```
python train_attribute_network.py
```

