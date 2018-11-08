import caffe
import sys
import tensorflow as tf
import pickle
import csv

def caffe2tf_step1(proto,weights,csvname):
    net = caffe.Net(proto,weights,caffe.TEST)
    f = open(csvname,'w')
    writer = csv.writer(f)
    print('This is all net params from caffe, remember names carefully!')

    dct = {}
    varlist = tf.trainable_variables()
    gList = tf.global_variables()
    for var in gList:
        # batch norm layer depends what you named in tf.
        if 'mu' in var.name or 'sigma' in var.name and var not in varlist:
            varlist.append(var)

    for var in varlist:
        writer.writerow([var.name,''])
    for key in net.params.keys():
        for i in range(len(net.params[key])):
            if i<2:
                writer.writerow([key+' %d'%i,''])
    f.close()

def caffe2tf_step2(proto,weights,csvname,pklname):
    f = open(csvname,'r')
    reader = csv.reader(f)
    lst = list(reader)
    net = caffe.Net(proto,weights,caffe.TEST)
    dct = {}
    for row in lst:
        if len(row[1])!=0:
            name = row[1].strip().split()[0]
            index = int(row[1].strip().split()[1])
            value=net.params[name][index].data
            if len(value.shape)==4:
                dct[row[0]]=value.transpose([2,3,1,0])
            elif len(value.shape)==2:
                dct[row[0]]=value.transpose()
            else:
                dct[row[0]]=value

    f.close()
    f = open(pklname,'wb')
    pickle.dump(dct,f)
    f.close()











if __name__=='__main__':
    '''
        tf model define
    '''
    from regressor4 import Regressor
    adam = tf.train.AdamOptimizer()
    model = Regressor(adam)
    with tf.variable_scope('model'):
        model.generate_model()


    #caffe2tf_step1('resnet_18_face_exp_test9.prototxt','solver3_drop9_iter_856000.caffemodel','map_.csv')
    caffe2tf_step2('resnet_18_face_exp_test9.prototxt','solver3_drop9_iter_856000.caffemodel','map_.csv','drop9.pkl')














