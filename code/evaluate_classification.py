##########################################################################################
# Import libraries and Path settings
##########################################################################################
import os
import matplotlib as mpl
if os.environ.get('DISPLAY','') == '':
    print('no display found. Using non-interactive Agg backend')
    mpl.use('Agg')

import matplotlib.pylab as plt
import numpy as np
import sys
from sklearn.metrics import roc_curve, auc, roc_auc_score, accuracy_score
from sklearn.metrics import precision_recall_curve, average_precision_score
os.environ['GLOG_minloglevel'] = '2' 
caffe_root = '/informatik2/students/home/4banik/Documents/caffe-master2/'
sys.path.insert(0, caffe_root + 'python')
import caffe
import os
from PIL import Image
import scipy.io as sio
from itertools import cycle
from scipy import interp

project_root = '../'
ia_data_root = '/informatik2/students/home/4banik/Documents/datasets/image_net_attribute/orig/'
#apascal_data_root = '/informatik2/students/home/4banik/Documents/datasets/VOC2008/JPEGImages/'
apascal_data_root='/home/soubarna/Workspace/datasets/apascal/JPEGImages/'
# ia_data_root = '/informatik2/students/home/4banik/Documents/datasets/image_net_attribute/cropped_to_bbox/'
# data_root = '/data_b/YCB_Video_Dataset/data/'
FONTSIZE=18

caffe.set_device(0)
caffe.set_mode_gpu()

##########################################################################################
# evaluation functions
##########################################################################################


def load_data(image_file, label_file, ds='IA', image_level=True):
    attribute_names = np.array(['black','blue','brown','gray','green','orange','pink','red','violet','white','yellow',
                            'long','rectangle','round','square',
                            'striped','spotted',
                            'furry','smooth','rough','shine','metal','veg','wood','wet'])
    if ds == 'IA':
        # Load attribute labels
        labels = np.loadtxt(label_file, delimiter=' ', dtype='float')
        print "Number of labels:", labels.shape
        print labels[0]

        # Load image names
        img_names = np.genfromtxt(image_file, dtype=None, usecols=0)
        print "Number of images: ", img_names.shape
        print img_names[0]
        attrib_index = range(25)
        return labels, img_names, attribute_names, attrib_index
    
    elif ds == 'apascal':
        # Load attribute labels
        labels = np.loadtxt(label_file,dtype=int)
        print 'Merging 2d boxy and 3d boxy (1st and 2nd column)'
        apascal_labels = np.empty((labels.shape[0], labels.shape[1]-1))
        apascal_labels[:, 0] = np.clip((labels[:, 0]+labels[:, 1]), 0, 1)
        apascal_labels[:, 1:] = labels[:, 2:]
        labels = apascal_labels
        
        # Load image names
        img_names = np.genfromtxt(image_file, dtype=None, usecols=0)
        print "Number of images: ", img_names.shape
        print img_names[0]
        
        # Load bbox list
        bbox_list = np.genfromtxt(image_file, usecols=(2, 3, 4, 5))
        u_img_labels = []
        if image_level:
            u_img_names = np.unique(img_names)
            for img in u_img_names:
                idx = np.where(img_names==img)
                temp_label = np.clip(np.sum(labels[idx], axis=0), 0, 1)
                u_img_labels.append(temp_label)
            u_img_labels=np.asarray(u_img_labels)
            img_names = u_img_names
            labels = u_img_labels
        
        # load attribute names
        apascal_attrib_index = [12,13,17,20,21,22,23]
        apascal_attribute_names = attribute_names[apascal_attrib_index]
        return labels, img_names, bbox_list, apascal_attribute_names, apascal_attrib_index
        
    elif ds == 'apascal_finetuned':
        # Load attribute labels
        labels = np.loadtxt(label_file, delimiter=' ', dtype='int')
        print "CKPT:"
        print "Number of labels:", labels.shape
        print labels[0]

        # Load image names
        img_names = np.genfromtxt(image_file, dtype=None, usecols=0)
        print "Number of images: ", img_names.shape
        print img_names[0]
        attrib_index = range(7)
        attrib_names = ['Rectangular','Round','Furry','Shiny','Metallic','Vegetation','Wooden']
        return labels, img_names, attrib_names, attrib_index
 


def evaluate_classification(net, transformer, image_file, label_file, plot_name='evaluate', dim=25, ds='IA', rerun=False, image_level=True):
    """
    evaluate classification performance
    net: model
    transformer: data pre-processer
    image_file: file with image names
    label_file: file with labels
    plot_name: output plot file name
    dim: number of output classes
    ds: dataset name
    rerun: if true run the model on images, else load saved predicted values from out dir
    """
    # set parameters
    new_size = (224, 224)
    it = 0
    y_score = []
    y_true = []
    err_cnt = 0
    err_list = []
    
    # Load data
    if ds == 'IA':
        labels, img_names, attribute_names, attrib_index = load_data(image_file, label_file, ds)
    elif ds == 'apascal':
        labels, img_names, bbox_list, attribute_names, attrib_index = load_data(image_file, label_file, ds)
    elif ds == 'apascal_finetuned':
        labels, img_names, attribute_names, attrib_index = load_data(image_file, label_file, ds)
        print img_names.shape
    
    total_images = img_names.shape[0]

    if not rerun:
        for img_name in img_names:
            try:
                image = Image.new('RGB', new_size)
                # Load image
                if ds == 'IA':
                    image = Image.open(ia_data_root + img_name)
                #elif ds == 'apascal':
                else:
                    if not labels[it].any:
                        it += 1
                        continue
                    image = Image.open(apascal_data_root + img_name)
                    if not image_level:
                        # obj level - crop till bbox of object
                        width, height = image.size
                        x1, y1, x2, y2 = bbox_list[it]
                        box_width = (x2 - x1)
                        box_height = (y2 - y1)
                        x1_ = max(0, (x1 - int(box_width * 0.1)))
                        x2_ = min(x1_ + box_width + int(2 * box_width * 0.1), width)
                        y1_ = max(0, y1 - int(box_height * 0.1))
                        y2_ = min(y1_ + box_height + int(2 * box_height * 0.1), height)
                        image = image.crop((x1_, y1_, x2_, y2_))
                
                # preprocess image
                image = image.resize(new_size, Image.ANTIALIAS)
                pil_image = np.asarray(image)
                transformed_image = transformer.preprocess('data', pil_image)
                net.blobs['data'].data[...] = transformed_image
                
                # predict output score
                out = net.forward()
                pred_probs = out['scores']
                pred_probs = pred_probs.reshape(dim, )
                pred_probs = pred_probs[attrib_index]
                
                # post-process score and gt                
                label = labels[it]
                gt = np.array(label >= 0.5).astype(int)
                y_true.append(gt)
                y_score.append(np.copy(pred_probs))

                if it % 1000 == 0:
                    print "Iteration: ", it, "/", total_images
                it += 1
            except Exception as e:
                print "Error in iteration ", it, " ", img_name, image.size, e.message, e.args
                err_cnt += 1
                # err_list.append("Error in iteration "+ str(it)+" "+img_name+" "+str(image.size)+" "+e.message)
                pass

        print "#Errors:", err_cnt
        print err_list
        np.savetxt(project_root + "log/" + plot_name + '_y_true.out', np.asarray(y_true).astype(int), delimiter=",", fmt='%i')
        np.savetxt(project_root + "log/" + plot_name + '_y_score.out', np.asarray(y_score), delimiter=",", fmt='%1.3f')
    else:
        y_true = np.loadtxt(project_root + "log/" + plot_name + '_y_true.out', delimiter=",", dtype=int)
        y_score = np.loadtxt(project_root + "log/" + plot_name + '_y_score.out', delimiter=",")

    print len(y_true), len(y_score)
    print y_true[0], y_score[0]
    
    # calculate roc, mAP
    if ds == 'IA':
        plot_ia(y_true, y_score, plot_name, dim, attribute_names, mode='micro')
        plot_ia(y_true, y_score, plot_name, dim, attribute_names, mode='macro')
    else:
        plot_apascal(y_true, y_score, plot_name, dim, attribute_names, mode='micro')
        plot_apascal(y_true, y_score, plot_name, dim, attribute_names, mode='macro')
    
    calc_accuracy(y_true, y_score, dim, attribute_names)


def plot_ia(y_true, y_score, plot_name, dim, attribute_names, mode='micro'):
    """
    calculates roc auc and mAP for dataset imagenet_attribute
    y_true: ground truth
    y_score: predicted scores
    plot_name: plot name
    dim: number of output class
    attribute_names: attribute names
    mode: micro/macro
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    color_true = y_true[:, :11]
    shape_true = y_true[:, 11:15]
    pattern_true = y_true[:, 15:17]
    texture_true = y_true[:, 17:]

    color_score = y_score[:, :11]
    shape_score = y_score[:, 11:15]
    pattern_score = y_score[:, 15:17]
    texture_score = y_score[:, 17:]

    # ROC AUC
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    fpr["all"], tpr["all"], _ = roc_curve(y_true.ravel(), y_score.ravel())
    fpr["color"], tpr["color"], _ = roc_curve(color_true.ravel(), color_score.ravel())
    fpr["shape"], tpr["shape"], _ = roc_curve(shape_true.ravel(), shape_score.ravel())
    fpr["pattern"], tpr["pattern"], _ = roc_curve(pattern_true.ravel(), pattern_score.ravel())
    fpr["texture"], tpr["texture"], _ = roc_curve(texture_true.ravel(), texture_score.ravel())
    roc_auc["all"] = roc_auc_score(y_true.ravel(), y_score.ravel(),average=mode)
    roc_auc["color"] = roc_auc_score(color_true.ravel(), color_score.ravel(), average=mode)
    roc_auc["shape"] = roc_auc_score(shape_true.ravel(), shape_score.ravel(), average=mode)
    roc_auc["pattern"] = roc_auc_score(pattern_true.ravel(), pattern_score.ravel(), average=mode)
    roc_auc["texture"] = roc_auc_score(texture_true.ravel(), texture_score.ravel(), average=mode)

    plt.figure(1)

    lines = []
    labels = []

    l, = plt.plot(fpr["all"], tpr["all"], color='black', linestyle='--', lw=2)
    lines.append(l)
    labels.append('All ({0:0.2f})'.format(roc_auc["all"]))

    l, = plt.plot(fpr["color"], tpr["color"], color='red')
    lines.append(l)
    labels.append('Color ({0:0.2f})'.format(roc_auc["color"]))

    l, = plt.plot(fpr["shape"], tpr["shape"], color='blue')
    lines.append(l)
    labels.append('Shape ({0:0.2f})'.format(roc_auc["shape"]))

    l, = plt.plot(fpr["pattern"], tpr["pattern"], color='orange')
    lines.append(l)
    labels.append('Pattern ({0:0.2f})'.format(roc_auc["pattern"]))

    l, = plt.plot(fpr["texture"], tpr["texture"], color='g')
    lines.append(l)
    labels.append('Texture ({0:0.2f})'.format(roc_auc["texture"]))

    fig = plt.gcf()
    fig.subplots_adjust(bottom=0.25, right=0.8)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=FONTSIZE)
    plt.ylabel('True Positive Rate', fontsize=FONTSIZE)
    # plt.title('Precision-Recall curve for - '+ plot_name)
    lgd = plt.legend(lines, labels, loc='lower right', bbox_to_anchor=(1, 0), ncol=1, fontsize=FONTSIZE-3)
    plt.savefig(project_root + 'plot/' + plot_name + '_' + mode + '_roc.eps', bbox_extra_artists=(lgd,), bbox_inches='tight')
    print "\nROC AUC - ", mode   
    print 'All attributes (ROC-AUC: {0:0.2f})'.format(roc_auc["all"])
    print 'Color (ROC-AUC: {0:0.2f})'.format(roc_auc["color"])
    print 'Shape (ROC-AUC: {0:0.2f})'.format(roc_auc["shape"])
    print 'Pattern (ROC-AUC: {0:0.2f})'.format(roc_auc["pattern"])
    print 'Texture (ROC-AUC: {0:0.2f})'.format(roc_auc["texture"])

    for i in range(dim):
        print "Class: ", attribute_names[i], " roc_auc = {0:0.2f}".format(roc_auc_score(y_true[:,i], y_score[:,i]))

    # Average Precision
    precision = dict()
    recall = dict()
    average_precision = dict()
    precision["all"], recall["all"], _ = precision_recall_curve(y_true.ravel(), y_score.ravel())
    precision["color"], recall["color"], _ = precision_recall_curve(color_true.ravel(), color_score.ravel())
    precision["shape"], recall["shape"], _ = precision_recall_curve(shape_true.ravel(), shape_score.ravel())
    precision["pattern"], recall["pattern"], _ = precision_recall_curve(pattern_true.ravel(), pattern_score.ravel())
    precision["texture"], recall["texture"], _ = precision_recall_curve(texture_true.ravel(), texture_score.ravel())
    average_precision["all"] = average_precision_score(y_true, y_score, average=mode)
    average_precision["color"] = average_precision_score(color_true, color_score, average=mode)
    average_precision["shape"] = average_precision_score(shape_true, shape_score, average=mode)
    average_precision["pattern"] = average_precision_score(pattern_true, pattern_score, average=mode)
    average_precision["texture"] = average_precision_score(texture_true, texture_score, average=mode)

    plt.figure(2)
    lines = []
    labels = []

    l, = plt.plot(recall["all"], precision["all"], color='black', linestyle='--', lw=2)
    lines.append(l)
    labels.append('All ({0:0.2f})'.format(average_precision["all"]))

    l, = plt.plot(recall["color"], precision["color"], color='red')
    lines.append(l)
    labels.append('Color ({0:0.2f})'.format(average_precision["color"]))

    l, = plt.plot(recall["shape"], precision["shape"], color='blue')
    lines.append(l)
    labels.append('Shape ({0:0.2f})'.format(average_precision["shape"]))

    l, = plt.plot(recall["pattern"], precision["pattern"], color='orange')
    lines.append(l)
    labels.append('Pattern ({0:0.2f})'.format(average_precision["pattern"]))

    l, = plt.plot(recall["texture"], precision["texture"], color='g')
    lines.append(l)
    labels.append('Texture ({0:0.2f})'.format(average_precision["texture"]))

    fig = plt.gcf()
    fig.subplots_adjust(bottom=0.25, right=0.8)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall', fontsize=FONTSIZE)
    plt.ylabel('Precision', fontsize=FONTSIZE)
    lgd = plt.legend(lines, labels, loc='lower left', bbox_to_anchor=(0, 0), ncol=1, fontsize=FONTSIZE - 3)
    plt.savefig(project_root + 'plot/' + plot_name + '_' + mode +  '_prec_recall.eps', bbox_extra_artists=(lgd,), bbox_inches='tight')
    # plt.show()

    attrib_avg_prec = np.zeros((dim), dtype=float)
    print "\nAverage Precision - ", mode
    print "All attributes ({0:0.2f})".format(average_precision["all"])
    for i in range(dim):
        attrib_avg_prec[i] = average_precision_score(y_true[:, i], y_score[:, i])
        print "Class: ", attribute_names[i], " AP = {0:0.2f}".format(attrib_avg_prec[i])

    plt.figure(3)
    fig, ax = plt.subplots()
    ind = np.arange(dim)

    ax.bar(range(11), attrib_avg_prec[:11], width=0.3, color='r',
           label='Color'.format(average_precision["color"]))
    ax.bar(ind[11:15], attrib_avg_prec[11:15], width=0.3, color='b',
           label='Shape'.format(average_precision["shape"]))
    ax.bar(ind[15:17], attrib_avg_prec[15:17], width=0.3, color='orange',
           label='Pattern'.format(average_precision["pattern"]))
    ax.bar(ind[17:], attrib_avg_prec[17:], width=0.3, color='g',
           label='Texture'.format(average_precision["texture"]))

    fig = plt.gcf()
    fig.subplots_adjust(bottom=0.25, right=0.9)
    plt.ylabel("Average precision \n (area under prec-recall curve)")
    plt.xlabel('Attribute')
    plt.xticks(ind, attribute_names, rotation=65, fontsize='medium', ha='right')
    plt.ylim([0.0, 1.05])
    lgd = plt.legend(loc=(0, -.38), ncol=4, fontsize='medium')
    ax.yaxis.grid(True)
    for line in ax.get_ygridlines():
        line.set_linestyle('-.')
        line.set_linewidth(0.20)
    plt.savefig(project_root + 'plot/category_' + plot_name + '_avg_prec.eps', bbox_extra_artists=(lgd,),
                bbox_inches='tight')


def plot_apascal(y_true, y_score, plot_name, dim, attribute_names, mode='micro'):
    """
    calculates roc auc and mAP for dataset apascal
    y_true: ground truth
    y_score: predicted scores
    plot_name: plot name
    dim: number of output class
    attribute_names: attribute names
    mode: micro/macro
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    shape_true = y_true[:, :3]
    texture_true = y_true[:, 3:]

    shape_score = y_score[:, :3]
    texture_score = y_score[:, 3:]
    
    # ROC AUC
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    fpr["all"], tpr["all"], _ = roc_curve(y_true.ravel(), y_score.ravel())
    fpr["shape"], tpr["shape"], _ = roc_curve(shape_true.ravel(), shape_score.ravel())
    fpr["texture"], tpr["texture"], _ = roc_curve(texture_true.ravel(), texture_score.ravel())
    roc_auc["all"] = roc_auc_score(y_true.ravel(), y_score.ravel(), average=mode)
    roc_auc["shape"] = roc_auc_score(shape_true.ravel(), shape_score.ravel(), average=mode)
    roc_auc["texture"] = roc_auc_score(texture_true.ravel(), texture_score.ravel(), average=mode)
    print "\nROC AUC - ", mode   
    print 'All attributes (ROC-AUC: {0:0.2f})'.format(roc_auc["all"])
    print 'Shape (ROC-AUC: {0:0.2f})'.format(roc_auc["shape"])
    print 'Texture (ROC-AUC: {0:0.2f})'.format(roc_auc["texture"])

    plt.figure(11)

    lines = []
    labels = []
    l, = plt.plot(fpr["all"], tpr["all"], color='black', linestyle='--', lw=2)
    lines.append(l)
    labels.append('All attributes ({0:0.2f})'.format(roc_auc["all"]))

    l, = plt.plot(fpr["shape"], tpr["shape"], color='blue')
    lines.append(l)
    labels.append('Shape ({0:0.2f})'.format(roc_auc["shape"]))

    l, = plt.plot(fpr["texture"], tpr["texture"], color='g')
    lines.append(l)
    labels.append('Texture ({0:0.2f})'.format(roc_auc["texture"]))

    fig = plt.gcf()
    fig.subplots_adjust(bottom=0.25, right=0.8)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=FONTSIZE)
    plt.ylabel('True Positive Rate', fontsize=FONTSIZE)
    lgd = plt.legend(lines, labels, loc='lower right', bbox_to_anchor=(1, 0), ncol=1, fontsize=FONTSIZE-3)
    plt.savefig(project_root + 'plot/' + plot_name + '_' + mode + '_roc.eps', bbox_extra_artists=(lgd,), bbox_inches='tight')

    # Average Precision
    precision = dict()
    recall = dict()
    average_precision = dict()
    precision["all"], recall["all"], _ = precision_recall_curve(y_true.ravel(), y_score.ravel())
    precision["shape"], recall["shape"], _ = precision_recall_curve(shape_true.ravel(), shape_score.ravel())
    precision["texture"], recall["texture"], _ = precision_recall_curve(texture_true.ravel(), texture_score.ravel())
    average_precision["all"] = average_precision_score(y_true, y_score, average=mode)
    average_precision["shape"] = average_precision_score(shape_true, shape_score, average=mode)
    average_precision["texture"] = average_precision_score(texture_true, texture_score, average=mode)

    plt.figure(22)

    lines = []
    labels = []
    l, = plt.plot(recall["all"], precision["all"], color='black', linestyle='--', lw=2)
    lines.append(l)
    labels.append('All attributes ({0:0.2f})'.format(average_precision["all"]))

    l, = plt.plot(recall["shape"], precision["shape"], color='blue')
    lines.append(l)
    labels.append('Shape ({0:0.2f})'.format(average_precision["shape"]))

    l, = plt.plot(recall["texture"], precision["texture"], color='g')
    lines.append(l)
    labels.append('Texture ({0:0.2f})'.format(average_precision["texture"]))

    fig = plt.gcf()
    fig.subplots_adjust(bottom=0.25, right=0.8)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall', fontsize=FONTSIZE)
    plt.ylabel('Precision', fontsize=FONTSIZE)
    lgd = plt.legend(lines, labels, loc='lower left', bbox_to_anchor=(0, 0), ncol=1, fontsize=FONTSIZE-3)
    plt.savefig(project_root + 'plot/' + plot_name + '_' + mode + '_prec_recall.eps', bbox_extra_artists=(lgd,), bbox_inches='tight')

    attrib_avg_prec = np.zeros(dim, dtype=float)
    print "\nAverage Precision - ", mode
    print "All attributes ({0:0.2f})".format(average_precision["all"])
    for i in range(dim):
        attrib_avg_prec[i] = average_precision_score(y_true[:, i], y_score[:, i])
        print "Class: ", attribute_names[i], " AP = {0:0.2f}".format(attrib_avg_prec[i])

    plt.figure(33)
    fig, ax = plt.subplots()
    ind = np.arange(dim)

    ax.bar(ind[:3], attrib_avg_prec[:3], width=0.3, color='b',
           label='Shape'.format(average_precision["shape"]))
    ax.bar(ind[3:], attrib_avg_prec[3:], width=0.3, color='g',
           label='Texture'.format(average_precision["texture"]))

    fig = plt.gcf()
    fig.subplots_adjust(bottom=0.25, right=0.9)
    plt.ylabel("Average precision \n (area under prec-recall curve)")
    plt.xlabel('Attribute')
    plt.xticks(ind, attribute_names, rotation=65, fontsize='medium', ha='right')
    plt.ylim([0.0, 1.05])
    lgd = plt.legend(loc=(0, -.38), ncol=4, fontsize='medium')
    ax.yaxis.grid(True)
    for line in ax.get_ygridlines():
        line.set_linestyle('-.')
        line.set_linewidth(0.20)
    plt.savefig(project_root + 'plot/category_' + plot_name + '_avg_prec.eps', bbox_extra_artists=(lgd,),
                bbox_inches='tight')


def calc_accuracy(y_true, y_score, dim, attribute_names):
    """
    calculates accuracy per class
    y_true: ground truth
    y_score: predicted scores
    dim: number of output class
    attribute_names: attribute names
    """
    y_true=np.asarray(y_true)
    y_score=np.asarray(y_score)
    y_score = (y_score >= 0.5).astype(int)

    print "\nAccuracy:"
    for i in range(dim):
        print "Class: ", attribute_names[i], " ACC = {0:0.2f}".format(accuracy_score(y_true[:,i], y_score[:,i]))


if __name__ == '__main__':
    # model_def = project_root + 'out/new_resnet.prototxt'
    # model_weights = project_root + 'models/resnet_caffe_trial11_iter_6000.caffemodel'

    # DAN weights
    model_def = '/informatik2/students/home/4banik/PycharmProjects/attribute_learning1/out/attribute_trial1_deploy.prototxt'
    model_weights = '/informatik2/students/home/4banik/PycharmProjects/attribute_learning1/models/attribute_cropped_trial21_iter_7000.caffemodel'
    #model_weights = '/data_b/soubarna/models/dan_cropped_adam_trial11_iter_8000.caffemodel'

    # CAM weights
    #model_def = project_root + 'out/attribute_camnw2_trial11_deploy.prototxt'
    #model_weights = '/data_b/soubarna/models/attribute_camnw2_trial11_iter_2000.caffemodel'
    
    # DAN finetuned apascal weights
    # model_def='/informatik2/students/home/4banik/workspace/attribute_attention/attribute_learning/out/dan_apascal_trial1_deploy.prototxt'
    # model_weights='/data_b/soubarna/models/dan_apascal_trial3/dan_apascal_trial32_iter_15000.caffemodel'   
     
    if os.path.isfile(model_weights):
        print 'model weights found.'

    net = caffe.Net(model_def,  # defines the structure of the model
                    model_weights,  # contains the trained weights
                    caffe.TEST)  # use test mode (e.g., don't perform dropout)

    # Transform input
    mu = np.load(caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy')
    mu = mu.mean(1).mean(1)  # average over pixels to obtain the mean (BGR) pixel values

    # VGG network expects input image in (C,H,W) format where C=(BGR)
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_transpose('data', (2, 0, 1))  # move image channels to outermost dimension
    transformer.set_mean('data', mu)  # subtract the dataset-mean value in each channel
    transformer.set_channel_swap('data', (2, 1, 0))  # required for PIL and caffe img, swap channels from RGB to BGR
    
    evaluate_classification(net, transformer, image_file=project_root + 'data/IA_X_test.txt',
              label_file=project_root + 'data/IA_Y_test.txt',
              plot_name='IA_TEST', dim=25, ds='IA', rerun=False)


    #evaluate_classification(net, transformer, image_file=project_root + 'data/apascal_test.txt',
    #                   label_file=project_root + 'data/apascal_test_labels_correct.txt',
    #                   plot_name='apascal_DAN', dim=25, ds='apascal', rerun=False, image_level=True)
    
    # evaluate_classification(net, transformer, image_file=project_root + 'data/apascal_X_val.txt',
    #                    label_file=project_root + 'data/apascal_Y_val.txt',
    #                    plot_name='apascal_dan_trial32_DA', dim=7, ds='apascal_finetuned', rerun=False, image_level=True)
