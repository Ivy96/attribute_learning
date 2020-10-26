import scipy.io as sio
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
import sys
import os
os.environ['GLOG_minloglevel'] = '2'
caffe_root = '/home/soubarna/Softwares/caffe-master/'
sys.path.insert(0, caffe_root + 'python')
import caffe
from PIL import Image as PImage
from PIL import ImageDraw
import argparse
import lmdb
from caffe.proto import caffe_pb2

#########################################################
# SETTINGS
#########################################################
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", required=True, help=" ia: imagenet_attrib else apascal")
parser.add_argument("--project_root",  default="../")
parser.add_argument("--data_root",  default="/home/soubarna/Workspace/datasets/atttribute_attention/datasets/imagenet_attrib/")
args = vars(parser.parse_args()) 
dataset = args['dataset']
project_root = args['project_root']
data_root = args['data_root']

print(dataset, project_root, data_root)
#########################################################
# DATA PROCESSING FUNCTIONS
#########################################################
def split_annotation():
    attrann = sio.loadmat(project_root + 'data/attrann.mat')['attrann'] 
    images_mat = attrann[0][0]['images']
    labels_mat = attrann[0][0]['labels']
    attributes_mat = attrann[0][0]['attributes']
    bboxes_mat = attrann[0][0]['bboxes']
    sio.savemat(project_root + 'data/mat_files/images.mat', {'images': images_mat})
    sio.savemat(project_root + 'data/mat_files/labels.mat', {'labels': labels_mat})
    sio.savemat(project_root + 'data/mat_files/attributes.mat', {'attributes': attributes_mat})
    sio.savemat(project_root + 'data/mat_files/bboxes.mat', {'bboxes': bboxes_mat})
    

def split_train_test_data(attrib_type, out_dir="data"):
    #########################################################
    # LOADING DATA
    #########################################################
    attributes_mat = sio.loadmat(project_root + 'data/mat_files/attributes.mat')
    attributes = attributes_mat['attributes']
    attributes = attributes.reshape(25)

    labels_mat = sio.loadmat(project_root + 'data/mat_files/labels.mat')
    labels = labels_mat['labels']

    images_mat = sio.loadmat(project_root + 'data/mat_files/images.mat')
    image_names = images_mat['images']
    image_names = image_names.reshape(9600)

    # Dividing labels to Attribute categories
    shape_attrib_index = [6, 10, 13, 17]
    pattern_attrib_index = [16, 18]
    tex_attrib_index = [3, 15, 12, 14, 7, 19, 23, 21]
    color_attrib_index = [0, 1, 2, 4, 5, 8, 9, 11, 20, 22, 24]

    attrib_index = []
    if attrib_type == 'shape':
        print "Shape attributes:"
        attrib_index =  shape_attrib_index
    elif attrib_type == 'pattern':
        print "Pattern attributes:"
        attrib_index =  pattern_attrib_index
    elif attrib_type == 'texture':
        print "Texture attributes:"
        attrib_index =  tex_attrib_index
    elif attrib_type == 'color':
        print "Color attributes:"
        attrib_index =  color_attrib_index
    elif attrib_type == 'all':
        print "all attributes"
        attrib_index = color_attrib_index + shape_attrib_index + pattern_attrib_index + tex_attrib_index

    print "attribute order color, shape, pattern, texture"
    print "Attributes:\n", attributes[attrib_index].ravel()
    master_labels = labels[:,attrib_index]
    print "Total label records: ", master_labels.shape

    image_names_edited = np.empty(image_names.shape, dtype='U40')
    for idx in range(len(image_names)):
        record = image_names[idx]
        image_names_edited[idx] = record[0].split('_')[0] + '/' + record[0] + '.JPEG 0'

    master_labels = master_labels.astype(float)
    master_labels[master_labels==0]=0.5
    master_labels[master_labels==-1]=0

    #########################################################
    # SPLIT TRAIN, VAL, TEST
    #########################################################
    X_train, X_test, y_train, y_test = train_test_split(image_names_edited, master_labels, test_size=0.20, random_state=42)
    X_train1, X_val, y_train1, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42)

    #########################################################
    # IMAGE_NAME, LABEL PRE-PROCESSING and SAVE TO FILE
    #########################################################
    if not os.path.exists(project_root+ out_dir):
        os.makedirs(project_root+ out_dir)
    
    file_path = project_root+ out_dir+'/ia_'+ attrib_type
    np.savetxt(file_path+'_X_train.txt', X_train1, fmt="%s")
    np.savetxt(file_path+'_X_val.txt', X_val, fmt="%s")
    np.savetxt(file_path+'_X_test.txt', X_test, fmt="%s")
    file_x_list = [file_path+'_X_train.txt', file_path+'_X_val.txt', file_path+'_X_test.txt']
    
    np.savetxt(file_path+'_Y_train.txt', y_train1, fmt="%.1f")
    np.savetxt(file_path+'_Y_val.txt', y_val, fmt="%.1f")
    np.savetxt(file_path+'_Y_test.txt', y_test, fmt="%.1f")
    file_y_list = [file_path+'_Y_train.txt', file_path+'_Y_val.txt', file_path+'_Y_test.txt']

    return file_x_list, file_y_list


def crop_to_object_bbox(data_root, file_x_val):
    """
    crops original images of imagenet dataset till bounding 
    box of the object. 
    """
    images_mat = sio.loadmat(project_root + 'data/mat_files/images.mat')
    image_names = images_mat['images']
    image_names = image_names.reshape(9600)

    bboxes_mat = sio.loadmat(project_root + 'data/mat_files/bboxes.mat')
    bboxes = bboxes_mat['bboxes']
    bboxes = bboxes.reshape(9600)

    # load training/validation imagelist
    imgList = np.genfromtxt(file_x_val, delimiter=' ', dtype=None, usecols=0)
    cnt = 0

    for imgName in imgList:
        if cnt % 100 == 0:
            print cnt, imgName

        # retrieve bbox
        itemindex=np.where(image_names==imgName.split('/')[1][:-5]) # img example n02100583/n02100583_10358.JPEG
        
        # load image
        img = PImage.open(data_root + 'orig/' + imgName)
        width, height = img.size
        
        bbox = np.array(list(bboxes[itemindex][0]),dtype=np.float).reshape(4,)
        x1 = int(bbox[0]*width)
        y1 = int(bbox[2]*height)
        x2 = int(bbox[1]*width)
        y2 = int(bbox[3]*height)

        # crop image
        box_width = (x2-x1)
        box_height = (y2-y1)
        x1_ = max(0, (x1 - int(box_width * 0.1)))
        x2_ = min(x1_ + box_width + int(2 * box_width * 0.1), width)
        y1_ = max(0, y1 - int(box_height * 0.1))
        y2_ = min(y1_ + box_height + int(2 * box_height * 0.1), height)
        crp_img = img.crop((x1_,y1_,x2_,y2_))

        # save image
        directory = os.path.dirname(data_root + 'cropped_to_bbox/' + imgName)
        if not os.path.exists(directory):
            os.makedirs(directory)

        crp_img.save(data_root + 'cropped_to_bbox/' + imgName)
        cnt +=1


def create_attrib_label_lmdb(label_file, lmdb_label_name, label_dim, out_dir="data"):
    """
    creates lmdb file for multilabels
    """
    Labels = list()
    print "Reading ", label_file, "..."
    Labels = np.loadtxt(label_file,  delimiter=' ')
    print "Records read - ", len(Labels), Labels[0]

    in_db_label = lmdb.open(project_root + out_dir + '/' + lmdb_label_name, map_size=int(1e12))
    with in_db_label.begin(write=True) as in_txn:
        for label_idx, label_ in enumerate(Labels):
            # datum = channels, width, height
            #im_dat = caffe.io.array_to_datum(np.array(label_).astype(float).reshape(label_dim,1,1)) #uncomment for imagenet_attrib label
            im_dat = caffe.io.array_to_datum(np.array(label_).astype(int).reshape(label_dim,1,1)) # uncomment for apascal
            in_txn.put('{:0>10d}'.format(label_idx), im_dat.SerializeToString())
            status = str(label_idx+1) + ' / ' + str(len(Labels))
            if label_idx % 1000 == 0:
                print status
    in_db_label.close()
    print lmdb_label_name, " created."

    # Verify lmdb

    read_flag = False
    if read_flag:
        print "Verify lmdb"
        lmdb_file = project_root + out_dir + '/' + lmdb_label_name
        lmdb_env = lmdb.open(lmdb_file)
        lmdb_txn = lmdb_env.begin()
        lmdb_cursor = lmdb_txn.cursor()
        datum = caffe_pb2.Datum()
        for key, value in lmdb_cursor:
            datum.ParseFromString(value)
            label = datum.label
            data = caffe.io.datum_to_array(datum)
            # im = data.astype(np.float)
            im = data.astype(np.int)
            # im = np.transpose(im, (2, 1, 0)) # original (dim, col, row)
            print "dummy label ", label, "/ label ", im
            break
        
        lmdb_env.close()


def prepare_apascal(ann_file, flag):
    object_classes = ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair','cow',
                'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor')
    class_to_ind = dict(zip(object_classes, xrange(20)))

    if flag== "train":
        # load image names
        imgList = np.genfromtxt(ann_file, dtype=None, usecols=0)
        print "Number of images: ", imgList.shape
        print imgList[0]  
        u_imgList = np.unique(imgList)
        
        # load label list
        attrib_index=np.asarray([0,1,2,56,52,61,62,54]) # [2d boxy, 3d boxy, round, furry, metallic, shiny, vegetation, wooden]
        labelList = np.genfromtxt(ann_file, usecols=(6+attrib_index))
        multilabelList = []
        
        # merge object level annotations to image level    
        for image_name in u_imgList:
            idx = np.where(imgList==image_name)
            temp_labels = labelList[idx]
            labels = np.empty([temp_labels.shape[0], temp_labels.shape[1]-1], dtype=int)
            # merge 2d  boxy and 3d boxy
            labels[:,0] = temp_labels[:,0]+temp_labels[:,1]
            labels[:,1:] = temp_labels[:,2:]
            multilabel = np.clip(np.sum(labels, axis=0), 0,1)
            multilabelList.append(multilabel)
        
        X = u_imgList    
        Y = np.asarray(multilabelList)
        print X.shape, Y.shape
        
        # split train to train- val
        X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.20, random_state=42)
        
        # save files
        np.savetxt('../data/apascal_X_train.txt', X_train, fmt="%s")
        np.savetxt('../data/apascal_Y_train.txt', Y_train, fmt="%i")
        np.savetxt('../data/apascal_X_val.txt', X_val, fmt="%s")
        np.savetxt('../data/apascal_Y_val.txt', Y_val, fmt="%i")

        # create label lmdb
        create_attrib_label_lmdb(label_file='../data/apascal_Y_train.txt', lmdb_label_name='apascal_Y_train_lmdb', label_dim=7)
        print("train labels lmdb created")
        create_attrib_label_lmdb(label_file='../data/apascal_Y_val.txt', lmdb_label_name='apascal_Y_val_lmdb', label_dim=7)
        print("validation labels lmdb created")

    elif flag == "test":
        imgList=np.genfromtxt(ann_file, delimiter=' ', dtype=None, usecols=(0))
        attrib_index=np.asarray([0,1,2,56,52,61,62,54])
        labelList=np.genfromtxt(ann_file, delimiter=' ', dtype=None, usecols=(6+attrib_index))
        u_imgList = np.unique(imgList)
        
        # attribute multilabels
        multilabelList=[]
        for image_name in u_imgList:
            idx = np.where(imgList==image_name)
            temp_labels = labelList[idx]
            labels = np.empty([temp_labels.shape[0], temp_labels.shape[1]-1], dtype=int)
            # merge 2d  boxy and 3d boxy
            labels[:,0] = temp_labels[:,0]+temp_labels[:,1]
            labels[:,1:] = temp_labels[:,2:]
            multilabel = np.clip(np.sum(labels, axis=0), 0,1)
            multilabelList.append(multilabel)

        X = u_imgList    
        Y = np.asarray(multilabelList)
        print X.shape, Y.shape

        np.savetxt('../data/apascal_X_test.txt', X, fmt="%s")
        np.savetxt('../data/apascal_Y_test.txt', Y, fmt="%i")
        
        # object multilabels
        obj_master_labels = np.genfromtxt(ann_file, delimiter=' ', dtype=None, usecols=(1))
        obj_multilabelList=[]
        for img in X:
            idx=np.where(imgList==img)
            multilabel = np.zeros(20).astype(int)
            anns = obj_master_labels[idx]
            print anns
            for ann in anns:
                cls_id = class_to_ind[ann]
                multilabel[cls_id]=1
            obj_multilabelList.append(multilabel)
            print multilabel

        obj_Y=np.asarray(obj_multilabelList)
        np.savetxt('../data/apascal_obj_Y_test.txt', obj_Y, fmt="%i")

if __name__ == '__main__':

    if dataset == "ia":
        out_dir="data2"
        # for imagenet attribute dataset
        # split into train/val/test
        file_x_split, file_y_split = split_train_test_data(attrib_type="all", out_dir=out_dir)
        print "\nAnnotation files:\nimage lists: ", file_x_split
        print "label lists: ", file_y_split, "\n"
        
        # crop to bbox for imagenet attrib
        print "Cropping images to bbox"
        for file_x in file_x_split:
            crop_to_object_bbox(data_root, file_x)
        
        # create label lmdb file
        print "\nCreating label lmdb files"
        for file_y in file_y_split:
            out_lmdb_filename = file_y.split("/")[-1].split(".")[0] + "_lmdb"
            create_attrib_label_lmdb(label_file=file_y, lmdb_label_name=out_lmdb_filename, label_dim=25, out_dir=out_dir)
    else:
        #prepare_apascal('/informatik2/students/home/4banik/workspace/attribute_attention/attribute_learning/data/apascal_train.txt')
        prepare_apascal('/informatik2/students/home/4banik/workspace/attribute_attention/attribute_learning/data/apascal_test.txt', 'test')
