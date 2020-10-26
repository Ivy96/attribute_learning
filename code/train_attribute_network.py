##########################################################################################
# Path settings
##########################################################################################
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--network_name", default="deep_attribute_network", help="network name")
parser.add_argument("--project_root",  default="../")
parser.add_argument("--data_root",  default="/home/soubarna/Workspace/datasets/atttribute_attention/datasets/")
parser.add_argument("--model_root",  default="/home/soubarna/Workspace/attribute_attention/attribute_learning/models/")
parser.add_argument("--caffe_root",  default='/home/soubarna/Softwares/caffe-master/')
parser.add_argument("--train_data_lmdb",  default='data2/ia_all_cropped_train_lmdb')
parser.add_argument("--train_label_lmdb",  default='data2/ia_all_Y_train_lmdb')
parser.add_argument("--val_data_lmdb",  default='data2/ia_all_cropped_val_lmdb')
parser.add_argument("--val_label_lmdb",  default='data2/ia_all_Y_val_lmdb')
parser.add_argument("--init_weight",  default='models/vgg_16/VGG_ILSVRC_16_layers.caffemodel', help="path to vgg16 caffemodel")
parser.add_argument("--restore_weight", default=None, help="path to solverstate")

args = vars(parser.parse_args()) 
network_name = args['network_name']
project_root = args['project_root']
data_root = args['data_root']
model_root = args['model_root']
caffe_root = args['caffe_root']
train_data_lmdb = args['train_data_lmdb']
val_data_lmdb = args['val_data_lmdb']
train_label_lmdb = args['train_label_lmdb']
val_label_lmdb = args['val_label_lmdb']
init_weight = args['init_weight']
restore_weight = args['restore_weight']

##########################################################################################
# Libraries
##########################################################################################

import numpy as np
import os, sys
from sklearn.metrics import accuracy_score
from six.moves import urllib
sys.path.insert(0, caffe_root + 'python')
import caffe
from caffe import layers as L
from caffe import params as P
sys.path.append(caffe_root + "examples/pycaffe/layers")
sys.path.append(caffe_root + "examples/pycaffe")
sys.path.append(caffe_root + "scripts")
import tools
from download_model_binary import reporthook
# caffe.set_device(0)
caffe.set_mode_cpu()

##########################################################################################
# Network parameters
##########################################################################################

weight_param = dict(lr_mult=1, decay_mult=1)
ft_weight_param = dict(lr_mult=10, decay_mult=1)
bias_param = dict(lr_mult=2, decay_mult=0)
ft_bias_param = dict(lr_mult=20, decay_mult=0)

learned_param = [weight_param, bias_param]
learned_param2 = [ft_weight_param, ft_bias_param]
frozen_param = [dict(lr_mult=0)] * 2

##########################################################################################
# Helper functions for building network
##########################################################################################


def conv_relu(bottom, ks, nout, stride=1, pad=0, group=1,
              param=learned_param,
              weight_filler=dict(type='gaussian', std=0.01),
              bias_filler=dict(type='constant', value=0.1)):
    conv = L.Convolution(bottom, kernel_size=ks, stride=stride,
                         num_output=nout, pad=pad, group=group,
                         param=param, weight_filler=weight_filler,
                         bias_filler=bias_filler)
    return conv, L.ReLU(conv, in_place=True)


def fc_relu(bottom, nout, param=learned_param,
            weight_filler=dict(type='gaussian', std=0.005),
            bias_filler=dict(type='constant', value=0.1)):
    fc = L.InnerProduct(bottom, num_output=nout, param=param,
                        weight_filler=weight_filler,
                        bias_filler=bias_filler)
    return fc, L.ReLU(fc, in_place=True)


def max_pool(bottom, ks, stride=1):
    return L.Pooling(bottom, pool=P.Pooling.MAX, kernel_size=ks, stride=stride)


##########################################################################################
# Network definition
##########################################################################################
def deep_attribute_network(output_net, train=True, num_classes=25, learn_all=False, batch_size=32):
    """
    Create attribute network from base network (VGG-16). 
    output_net: output file name
    train: True for training, False for testing
    num_classes: number of output classes
    learn_all: flag for learning all layers
    """

    n = caffe.NetSpec()
    if train:
        n.data, n.dummy_label1 = L.Data(batch_size=batch_size, backend=P.Data.LMDB,
                                        source=project_root + train_data_lmdb,
                                        transform_param=dict(mean_value=[104, 117, 123],
                                                             mirror=True, crop_size=224, scale= 0.00390625),
                                        ntop=2)
        n.label_1, n.dummy_label2 = L.Data(batch_size=batch_size, backend=P.Data.LMDB,
                                           source=project_root + train_label_lmdb,
                                           ntop=2)
        n.label = L.Flatten(n.label_1)
        n.silence = L.Silence(n.dummy_label1, n.dummy_label2, ntop=0)
    else:
        n.data, n.dummy_label1 = L.Data(batch_size=batch_size, backend=P.Data.LMDB,
                                        source=project_root + val_data_lmdb,
                                        transform_param=dict(mean_value=[104, 117, 123],
                                                             mirror=False, crop_size=224, scale= 0.00390625),
                                        ntop=2)
        n.label_1, n.dummy_label2 = L.Data(batch_size=batch_size, backend=P.Data.LMDB,
                                           source=project_root + val_label_lmdb,
                                           ntop=2)
        n.label = L.Flatten(n.label_1)
        n.silence = L.Silence(n.dummy_label1, n.dummy_label2, ntop=0)

    param = learned_param if learn_all else frozen_param
    n.conv1_1, n.relu1_1 = conv_relu(n.data, 3, 64, pad=1, param=param)
    n.conv1_2, n.relu1_2 = conv_relu(n.relu1_1, 3, 64, pad=1, param=param)
    n.pool1 = max_pool(n.relu1_2, 2, stride=2)
    n.conv2_1, n.relu2_1 = conv_relu(n.pool1, 3, 128, pad=1, param=param)
    n.conv2_2, n.relu2_2 = conv_relu(n.relu2_1, 3, 128, pad=1, param=param)
    n.pool2 = max_pool(n.relu2_2, 2, stride=2)
    n.conv3_1, n.relu3_1 = conv_relu(n.pool2, 3, 256, pad=1, param=param)
    n.conv3_2, n.relu3_2 = conv_relu(n.relu3_1, 3, 256, pad=1, param=param)
    n.conv3_3, n.relu3_3 = conv_relu(n.relu3_2, 3, 256, pad=1, param=param)
    n.pool3 = max_pool(n.relu3_3, 2, stride=2)
    n.conv4_1, n.relu4_1 = conv_relu(n.pool3, 3, 512, pad=1, param=param)
    n.conv4_2, n.relu4_2 = conv_relu(n.relu4_1, 3, 512, pad=1, param=param)
    n.conv4_3, n.relu4_3 = conv_relu(n.relu4_2, 3, 512, pad=1, param=param)
    n.pool4 = max_pool(n.relu4_3, 2, stride=2)
    n.conv5_1, n.relu5_1 = conv_relu(n.pool4, 3, 512, pad=1, param=learned_param)
    n.conv5_2, n.relu5_2 = conv_relu(n.relu5_1, 3, 512, pad=1, param=learned_param)
    n.conv5_3, n.relu5_3 = conv_relu(n.relu5_2, 3, 512, pad=1, param=learned_param)
    n.pool5 = max_pool(n.relu5_3, 2, stride=2)

    n.fc6, n.relu6 = fc_relu(n.pool5, 4096, param=learned_param)
    if train:
        n.drop6 = fc7input = L.Dropout(n.relu6, in_place=True)
    else:
        fc7input = n.relu6

    n.fc7_, n.relu7_ = fc_relu(fc7input, 1024, param=learned_param)
    if train:
        n.drop7_ = fc8input = L.Dropout(n.relu7_, in_place=True)
    else:
        fc8input = n.relu7_

    n.fc8_ = L.InnerProduct(fc8input, num_output=num_classes, param=learned_param)

    if not train:
        n.scores = L.Sigmoid(n.fc8_)
        n.accuracy = L.MultiLabelAccuracy(n.scores, n.label)

    n.class_weights = L.LossWeight(n.label)
    n.loss = L.WeightedSigmoidCrossEntropyLoss(n.fc8_, n.label, n.class_weights)

    f = open(project_root + 'out/' + output_net, 'w')
    f.write(str(n.to_proto()))
    f.close()


def attribute_cam_network(output_net, train=True, num_classes=25, learn_all=False, batch_size=32):
    """
        creates attribute_cam_network from deep_attribute_network.
        Follows recommended architecture from paper "Learning Deep Features for
        Discriminative Localization" (https://goo.gl/vWgH3w)
    """
    n = caffe.NetSpec()
    if train:
        n.data, n.dummy_label1 = L.Data(batch_size=batch_size, backend=P.Data.LMDB,
                                        source=project_root + train_data_lmdb,
                                        transform_param=dict(mean_value=[104, 117, 123],
                                                             mirror=True, crop_size=224),
                                        ntop=2)
        n.label_1, n.dummy_label2 = L.Data(batch_size=batch_size, backend=P.Data.LMDB,
                                           source=project_root + train_label_lmdb,
                                           ntop=2)
        n.label = L.Flatten(n.label_1)
        n.silence = L.Silence(n.dummy_label1, n.dummy_label2, ntop=0)
    else:
        n.data, n.dummy_label1 = L.Data(batch_size=batch_size, backend=P.Data.LMDB,
                                        source=project_root + val_data_lmdb,
                                        transform_param=dict(mean_value=[104, 117, 123],
                                                             mirror=False, crop_size=224),
                                        ntop=2)
        n.label_1, n.dummy_label2 = L.Data(batch_size=batch_size, backend=P.Data.LMDB,
                                           source=project_root + val_label_lmdb,
                                           ntop=2)
        n.label = L.Flatten(n.label_1)
        n.silence = L.Silence(n.dummy_label1, n.dummy_label2, ntop=0)

    param = learned_param if learn_all else frozen_param
    n.conv1_1, n.relu1_1 = conv_relu(n.data, 3, 64, pad=1, param=param)
    n.conv1_2, n.relu1_2 = conv_relu(n.relu1_1, 3, 64, pad=1, param=param)
    n.pool1 = max_pool(n.relu1_2, 2, stride=2)
    n.conv2_1, n.relu2_1 = conv_relu(n.pool1, 3, 128, pad=1, param=param)
    n.conv2_2, n.relu2_2 = conv_relu(n.relu2_1, 3, 128, pad=1, param=param)
    n.pool2 = max_pool(n.relu2_2, 2, stride=2)
    n.conv3_1, n.relu3_1 = conv_relu(n.pool2, 3, 256, pad=1, param=param)
    n.conv3_2, n.relu3_2 = conv_relu(n.relu3_1, 3, 256, pad=1, param=param)
    n.conv3_3, n.relu3_3 = conv_relu(n.relu3_2, 3, 256, pad=1, param=param)
    n.pool3 = max_pool(n.relu3_3, 2, stride=2)
    n.conv4_1, n.relu4_1 = conv_relu(n.pool3, 3, 512, pad=1, param=param)
    n.conv4_2, n.relu4_2 = conv_relu(n.relu4_1, 3, 512, pad=1, param=param)
    n.conv4_3, n.relu4_3 = conv_relu(n.relu4_2, 3, 512, pad=1, param=param)
    n.pool4 = max_pool(n.relu4_3, 2, stride=2)
    n.conv5_1, n.relu5_1 = conv_relu(n.pool4, 3, 512, pad=1, param=param)
    n.conv5_2, n.relu5_2 = conv_relu(n.relu5_1, 3, 512, pad=1, param=param)
    n.conv5_3, n.relu5_3 = conv_relu(n.relu5_2, 3, 512, pad=1, param=param)

    n.conv6 = L.Convolution(n.relu5_3, kernel_size=3, stride=1,
                            num_output=512, pad=1, group=1,
                            param=learned_param, weight_filler=dict(type='gaussian', std=0.01),
                            bias_filler=dict(type='constant', value=0.1))

    n.gap7 = L.Pooling(n.conv6, pool=P.Pooling.AVE, global_pooling=True)

    n.fc8_ = L.InnerProduct(n.gap7, num_output=num_classes, param=learned_param)

    if not train:
        n.scores = L.Sigmoid(n.fc8_)
        n.accuracy = L.MultiLabelAccuracy(n.scores, n.label)

    n.class_weights = L.LossWeight(n.label)
    n.loss = L.WeightedSigmoidCrossEntropyLoss(n.fc8_, n.label, n.class_weights)

    f = open(project_root + 'out/' + output_net, 'w')
    f.write(str(n.to_proto()))
    f.close()


def dan_apascal(output_net, train=True, num_classes=7, learn_all=False):
    """
    Create attribute network from base network (VGG-16). 
    output_net: output file name
    train: True for training, False for testing
    num_classes: number of output classes
    learn_all: flag for learning all layers
    """
    n = caffe.NetSpec()
    if train:
        n.data, n.dummy_label1 = L.Data(batch_size=32, backend=P.Data.LMDB,
                                        source='/data_b/soubarna/data/X256_apascal_train_lmdb',
                                        transform_param=dict(mean_value=[104, 117, 123],
                                                             mirror=True, crop_size=224, scale= 0.00390625),
                                        ntop=2)
        n.label_1, n.dummy_label2 = L.Data(batch_size=32, backend=P.Data.LMDB,
                                           source=project_root + 'data/apascal_Y_train_lmdb',
                                           ntop=2)
        n.label = L.Flatten(n.label_1)
        n.silence = L.Silence(n.dummy_label1, n.dummy_label2, ntop=0)
    else:
        n.data, n.dummy_label1 = L.Data(batch_size=32, backend=P.Data.LMDB,
                                        source='/data_b/soubarna/data/X256_apascal_val_lmdb',
                                        transform_param=dict(mean_value=[104, 117, 123],
                                                             mirror=False, crop_size=224, scale= 0.00390625),
                                        ntop=2)
        n.label_1, n.dummy_label2 = L.Data(batch_size=32, backend=P.Data.LMDB,
                                           source=project_root + 'data/apascal_Y_val_lmdb',
                                           ntop=2)
        n.label = L.Flatten(n.label_1)
        n.silence = L.Silence(n.dummy_label1, n.dummy_label2, ntop=0)

    param = learned_param if learn_all else frozen_param
    n.conv1_1, n.relu1_1 = conv_relu(n.data, 3, 64, pad=1, param=param)
    n.conv1_2, n.relu1_2 = conv_relu(n.relu1_1, 3, 64, pad=1, param=param)
    n.pool1 = max_pool(n.relu1_2, 2, stride=2)
    n.conv2_1, n.relu2_1 = conv_relu(n.pool1, 3, 128, pad=1, param=param)
    n.conv2_2, n.relu2_2 = conv_relu(n.relu2_1, 3, 128, pad=1, param=param)
    n.pool2 = max_pool(n.relu2_2, 2, stride=2)
    n.conv3_1, n.relu3_1 = conv_relu(n.pool2, 3, 256, pad=1, param=param)
    n.conv3_2, n.relu3_2 = conv_relu(n.relu3_1, 3, 256, pad=1, param=param)
    n.conv3_3, n.relu3_3 = conv_relu(n.relu3_2, 3, 256, pad=1, param=param)
    n.pool3 = max_pool(n.relu3_3, 2, stride=2)
    n.conv4_1, n.relu4_1 = conv_relu(n.pool3, 3, 512, pad=1, param=param)
    n.conv4_2, n.relu4_2 = conv_relu(n.relu4_1, 3, 512, pad=1, param=param)
    n.conv4_3, n.relu4_3 = conv_relu(n.relu4_2, 3, 512, pad=1, param=param)
    n.pool4 = max_pool(n.relu4_3, 2, stride=2)
    n.conv5_1, n.relu5_1 = conv_relu(n.pool4, 3, 512, pad=1, param=param)
    n.conv5_2, n.relu5_2 = conv_relu(n.relu5_1, 3, 512, pad=1, param=param)
    n.conv5_3, n.relu5_3 = conv_relu(n.relu5_2, 3, 512, pad=1, param=param)
    n.pool5 = max_pool(n.relu5_3, 2, stride=2)

    n.fc6, n.relu6 = fc_relu(n.pool5, 4096, param=learned_param)
    if train:
        n.drop6 = fc7input = L.Dropout(n.relu6, in_place=True)
    else:
        fc7input = n.relu6

    n.fc7_, n.relu7_ = fc_relu(fc7input, 1024, param=learned_param)
    if train:
        n.drop7_ = fc8input = L.Dropout(n.relu7_, in_place=True)
    else:
        fc8input = n.relu7_

    n.fc8 = L.InnerProduct(fc8input, num_output=num_classes, param=learned_param)

    if not train:
        n.scores = L.Sigmoid(n.fc8)
        n.accuracy = L.MultiLabelAccuracy(n.scores, n.label)

    #n.class_weights = L.LossWeight(n.label)
    n.loss = L.SigmoidCrossEntropyLoss(n.fc8, n.label)

    f = open(project_root + 'out/' + output_net, 'w')
    f.write(str(n.to_proto()))
    f.close()


def run_solver(solver, start_niter, niter, test_interval, n_test_iter, display_interval, num_classes, log_name, source_weight):
    print "Training started." 
    train_loss = np.zeros((niter/display_interval, 2))
    test_loss_acc = np.zeros((int(np.ceil(niter / test_interval)),num_classes+3))
    # the main solver loop
    for it in range(start_niter, niter):
        print "Iteration: ", it
        solver.step(1)  # SGD by Caffe

        # train loss
        if it % display_interval == 0:
            train_loss[it/display_interval][0] = it
            train_loss[it/display_interval][1] = solver.net.blobs['loss'].data

        # test loss and accuracy
        if it % test_interval == 0:
            print 'Iteration', it, 'testing...'
            acc = np.zeros(num_classes+1)
            loss = 0.0
            for test_it in range(n_test_iter):
                solver.test_nets[0].forward()
                loss += solver.test_nets[0].blobs['loss'].data
                y_pred = (solver.test_nets[0].blobs['scores'].data>=0.5).astype(int)
                y_true = (solver.test_nets[0].blobs['label'].data>=0.5).astype(int)
                # mean accuracy
                acc[0] += accuracy_score(y_true, y_pred) 
                # accuracy per attribute class   
                for i in range(0, num_classes):
                    acc[i+1] += accuracy_score(y_true[:,i], y_pred[:,i])

            test_loss_acc[it // test_interval][0] = it
            test_loss_acc[it // test_interval][1] = loss/n_test_iter
            for i in range(0,num_classes+1):
                test_loss_acc[it // test_interval][i+2] = acc[i]/n_test_iter

            print "Test accuracy: Total - class specific"
            print test_loss_acc[it // test_interval][2:]

    print "training loss len: :", len(train_loss)
    print "test loss len: ", len(test_loss_acc)
    # save loss, acc to csv
    np.savetxt(project_root+"log/train_"+log_name, train_loss, delimiter=",", fmt='%1.2f')
    np.savetxt(project_root+"log/test_"+log_name, test_loss_acc, delimiter=",", fmt='%1.2f')
    print "run_name: ", run_name, " complete"
    print "restored from: ", source_weight
    print "Logs: train/test_", log_name


##########################################################################################
# Call solver
##########################################################################################

if network_name == 'deep_attribute_network':
    #run_name = 'attribute_cropped_trial1'
    run_name = 'dan_cropped_adam_trial5'
    num_training_samples = 5760
    num_val_samples = 1920
    batch_size = 32
    num_epochs = 1

    # generate train/test prototxt
    deep_attribute_network(output_net=run_name+'_train.prototxt', train=True, num_classes=25, batch_size=batch_size)
    deep_attribute_network(output_net=run_name+'_test.prototxt', train=False, num_classes=25, batch_size=batch_size)
    print 'Network specification generated.'

    # generate solver
    solverprototxt = tools.CaffeSolver(
        trainnet_prototxt_path=project_root + 'out/'+run_name+'_train.prototxt',
        testnet_prototxt_path=project_root + 'out/'+run_name+'_test.prototxt')
    solverprototxt.sp['test_iter'] = str(num_val_samples/batch_size)
    solverprototxt.sp['test_interval'] = "500"
    #solverprototxt.sp['base_lr'] = "0.001"
    solverprototxt.sp['momentum'] = "0.9"
    #solverprototxt.sp['weight_decay'] = "0.0005"
    #solverprototxt.sp['lr_policy'] = "'step'"
    #solverprototxt.sp['type'] = "'SGD'"
    #solverprototxt.sp['gamma'] = "0.1"
    #solverprototxt.sp['stepsize'] = "5000"
    solverprototxt.sp['display'] = "100"
    solverprototxt.sp['max_iter'] = str(num_epochs * num_training_samples/batch_size)
    solverprototxt.sp['snapshot'] = "3000"
    solverprototxt.sp['random_seed'] = "147"
    solverprototxt.sp['snapshot_prefix'] = "'" + model_root + "dan_adam/" + run_name + "'"
    solverprototxt.sp['solver_mode'] = "GPU"
    # Adam solver settings    
    solverprototxt.sp['type'] = "'Adam'"
    solverprototxt.sp['momentum2'] = "0.999"
    solverprototxt.sp['lr_policy'] = "'fixed'"
    solverprototxt.sp['base_lr'] = "0.001"
    solverprototxt.write(project_root + 'out/' + run_name + '.prototxt')

    # load solver
    #solver = caffe.SGDSolver(project_root + 'out/' + run_name + '.prototxt')
    solver = caffe.AdamSolver(project_root + 'out/' + run_name + '.prototxt')
    
    # pretrained weight
    model_weight = ""
    if init_weight:
        pretrained_weight = caffe_root + init_weight
        model_weight = pretrained_weight
        pretrained_weight_dir = os.path.dirname(pretrained_weight)

        if os.path.isfile(pretrained_weight):
            print 'Pretrained weight VGG16 found.'
        else:
            # Download and verify model.
            print 'Downloading pre-trained VGG16 model...'
            vgg16_url = "http://www.robots.ox.ac.uk/~vgg/software/very_deep/caffe/VGG_ILSVRC_16_layers.caffemodel"
            if not os.path.exists(pretrained_weight_dir):
                os.makedirs(pretrained_weight_dir)
            urllib.request.urlretrieve(vgg16_url, pretrained_weight, reporthook)
            if not os.path.isfile(pretrained_weight):
                print 'Error in downloading. Download vgg16 from ', vgg16_url
                exit(1)
            solver.net.copy_from(pretrained_weight)

    # For continuing training
    if restore_weight:
        # model_weight = model_root + "models/dan_cropped_adam_trial43_iter_30000.solverstate"
        model_weight = model_root + restore_weight
        solver.restore(model_weight)

    # solver run parameters
    start_niter = 0
    niter = num_epochs * num_training_samples / batch_size
    test_interval = 500
    n_test_iter = num_val_samples / batch_size
    display_interval = 100
    num_classes = 25

    # log variables
    log_name = run_name + '.log'
    print "Num of epoch: ", num_epochs
    print "Num of iterations: ", niter
    run_solver(solver, start_niter, niter, test_interval, n_test_iter, display_interval, num_classes, log_name, model_weight)

