# Ricardo Sousa
# rsousa at rsousa.org

# Copyright 2014 Ricardo Sousa

# This file is part of NanoParticles.

# NanoParticles is free software: you can redistribute it and/or modify 
# it under the terms of the GNU General Public License as published
# by the Free Software Foundation, either version 3 of the License,
# or (at your option) any later version.

# NanoParticles is distributed in the hope that it will be useful, but 
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU 
# General Public License for more details.

# You should have received a copy of the GNU General Public License 
# along with NanoParticles. If not, see http://www.gnu.org/licenses/.

# ------------------------------------------------------------------------------------

import csv, re, sys, string, gzip, cPickle
import numpy, h5py

import theano
import theano.tensor as T

import cv2
from os import listdir, system
from os.path import isfile, join, splitext

from matplotlib import pyplot as plt

import cPickle as pickle

from data_handling import save_results, save_gzdata, load_savedgzdata

def confusion_matrix(ytest,ypred,K):
    nsamples = min(len(ytest),len(ypred))
    
    cm = numpy.zeros((K,K))
    for elem in range(0,nsamples):
        cm[ytest[elem],ypred[elem]] = cm[ytest[elem],ypred[elem]] + 1

    return cm

def view_data( data, label ):
    (npoints, ndim) = data.shape

    imgwidth = numpy.sqrt( ndim )
    
    for np in xrange(0,npoints):
        p = data[np,:]*255
        p.resize((imgwidth,imgwidth))

        if label[np] == 1:
            plt.imshow(p,cmap='gray')
            plt.show()


def loadConvertMNIST(patchsize, ids, dataset, imgwidth, train_set_x, train_set_y):
    nelem = len(train_set_y)
    for k in range(0,nelem):
        p = train_set_x[k,:]
        p = numpy.reshape(p,(imgwidth,imgwidth))
        p = cv2.resize(p,(patchsize,patchsize))
        p = p.ravel()
        # view_data(p, train_set_y[k])
        dataset[:,ids] = numpy.r_[ids,p,train_set_y[k]]
        ids = ids + 1

    return (dataset, ids)

def load_data(datasetpath, options):
    ''' Loads the dataset

    :type dataset: string
    :param dataset: the path to the dataset 
    '''

    # if options['oneclass']:
    #     nclasses = 1
    # else:
    #     nclasses = 2

    nclasses = 2
    
    #############
    # LOAD DATA #
    #############

    if options['database'] == 'mnist':
        train_set, valid_set, test_set = load_savedgzdata('mnist.pkl.gz')

        train_set_x = numpy.array(train_set[0])
        train_set_y = numpy.array(train_set[1])
        valid_set_x = numpy.array(valid_set[0])
        valid_set_y = numpy.array(valid_set[1])
        test_set_x  = numpy.array(test_set[0])
        test_set_y  = numpy.array(test_set[1])

        (nelem_train,ndim) = train_set_x.shape
        (nelem_valid,ndim) = valid_set_x.shape
        (nelem_test,ndim)  = test_set_x.shape
        
        dataset  = numpy.zeros((options['patchsize']*options['patchsize']+2,nelem_train+nelem_valid+nelem_test))
        ids      = 0
        imgwidth = numpy.sqrt(ndim)
        #print >> sys.stderr, "train....", ids
        (dataset, ids) = loadConvertMNIST(options['patchsize'], ids, dataset, imgwidth, train_set_x, train_set_y)
        #print >> sys.stderr, "val....", ids
        (dataset, ids) = loadConvertMNIST(options['patchsize'], ids, dataset, imgwidth, valid_set_x, valid_set_y)
        #print >> sys.stderr, "test....", ids
        (dataset, ids) = loadConvertMNIST(options['patchsize'], ids, dataset, imgwidth, test_set_x, test_set_y )
        
        nclasses = len(list(set(train_set_y)))
        return (dataset, options['patchsize']*options['patchsize'], nclasses)

    elif options['database'] == 'shapes':
        dataset = load_savedgzdata('shapes.pkl.gz')

        return(dataset,20*20,4)
        
    # --------------------------------------------------------------
    # data_dir, data_file = os.path.split(dataset)
    if not 'all' in options['database']:
        onlyfiles = [ f for f in listdir(datasetpath) \
                      if ( isfile(join(datasetpath,f)) and splitext(f)[1] == '.mat' and \
                           options['database'] in f and options['resolution'] in f ) ]
    else:
        onlyfiles = [ f for f in listdir(datasetpath) if ( isfile(join(datasetpath,f)) and splitext(f)[1] == '.mat' ) ]

    # onlyfiles = [ f for f in listdir(datasetpath) if ( isfile(join(datasetpath,f)) and splitext(f)[1] == '.mat' ) ]
    onlyfiles.sort()

    first = True
    ids   = 0;
    for file in onlyfiles:
        print >> sys.stderr, ( "---> " + datasetpath + file )
        f = h5py.File(datasetpath + file,'r')
        
        # print f.items();
        mpatches = f.get('mpatches')
        # print mpatches.items();
        back = numpy.array( mpatches.get('negative') )
        nano = numpy.array( mpatches.get('positive') )
        if options['replicate']:
            nano = numpy.c_[ back, back ]
        
        # print >> sys.stderr, back.shape
        # print >> sys.stderr, nano.shape
        
        (back_ndim,back_npoints) = back.shape
        (nano_ndim,nano_npoints) = nano.shape
        back[0,:] = back[0,:] + ids
        nano[0,:] = nano[0,:] + ids

        ids = max(back[0,:])
        # raw_input('> press any key <')
        
        back = numpy.r_[ back, numpy.ones((1,back_npoints)) ]
        nano = numpy.r_[ nano, -1 * numpy.ones((1,nano_npoints)) ]

        if first:
            dataset = numpy.c_[ back, nano]
            first   = False
        else:
            dataset = numpy.c_[ dataset, back, nano]
        
        # print type( dataset )
        # raw_input('....')

    # dataset was constructed according to the following structure
    # [[ .... ids ....],
    #  [ .... data ...],
    #  [ ..... cls ...]]

    datasetfilename = 'nanoparticles.npz'
    #save_gzdata(datasetfilename,dataset)
    numpy.savez_compressed(datasetfilename,dataset)

    kakak
    return (dataset, nano_ndim-1, nclasses)

def get_data( dataset, ids, options, isFirst = True, minvalue=0, maxvalue=1 ):

    # get idx from id's
    idx = numpy.in1d(dataset[0,:],ids)

    if isFirst == False and options['oneclass'] == True:
        # print dataset.shape
        # print idx.shape
        # remove background
        ind = dataset[-1,:] == 1
        # remove negative examples
        idx[ind] = False
        
    # get all data except ids and classes
    train_set_x = dataset[1:-1,idx].T
    # convert classes for 0,1,...,K
    if options['datanormalize']:
        train_set_y = (dataset[-1,idx] + 1 ) / 2
    else:
        train_set_y = dataset[-1,idx]
        
    if isFirst:
        minvalue = 0 #numpy.min( train_set_x )
        maxvalue = 255 #numpy.max( train_set_x )

    # print "Min value: {0:0.2f} | Max value: {1:.2f}".format( minvalue, maxvalue )
    if options['datanormalize']:
        train_set_x = (train_set_x - minvalue) / (maxvalue-minvalue+0.001)
    
    x, y = shared_dataset(train_set_x,train_set_y)
    
    return (x,y,minvalue,maxvalue)


def gen_folds( dataset, options, nrun ):
    nids = len( set( dataset[0,:] ) )
    ids  = options['numpy_rng'].permutation(nids)

    # train / test ids
    trainsizeElem = round( options['trainsize']*nids )
    train_ids = ids[0:trainsizeElem]    
    test_ids  = ids[trainsizeElem+1:nids]

    print >> sys.stderr, test_ids   

    if options['verbose']> 2:
        print >> sys.stderr, "Train IDS"
        print >> sys.stderr, train_ids
        print >> sys.stderr, "Test IDS"
        print >> sys.stderr, test_ids

    # val ids
    val_ids   = numpy.copy(train_ids)
    nitems    = len(val_ids)/options['folds']

    val_ids.resize((options['folds'],nitems))

    folds = range(0,options['folds'])

    trainval = []
    valval   = []
    testval  = []
    for k in folds:
        others = list( set([k]).symmetric_difference(set(folds)) )
        #print val_ids
        #print others
        #kk

        train  = val_ids[k].flatten()
        val    = val_ids[others[0]]
        test   = val_ids[others[1]]

        xtrain,ytrain,minv,maxv = get_data( dataset, train, options )
        xval,yval   = get_data( dataset,  val, options, isFirst=False, minvalue=minv, maxvalue=maxv )[0:2]
        xtest,ytest = get_data( dataset, test, options, isFirst=False, minvalue=minv, maxvalue=maxv )[0:2]
        
        trainval.append( (xtrain,ytrain) )
        valval.append( (xval,yval) )
        testval.append( (xtest,ytest) )

        if options['verbose'] > 0:
            print 'Train set with size %d for fold %d' % (ytrain.shape.eval(),k)
            print 'Test  set with size %d for fold %d' % (ytest.shape.eval(),k)
            if options['verbose'] > 5:
                for cls in range(0,2):
                    print >>sys.stderr, "\tNumber of training elements for cls {0:02d} is {1:05d}".format(cls,sum(ytrain.eval() == cls))
                    print >>sys.stderr, "\tNumber of testing elements for cls {0:02d} is {1:05d}".format(cls,sum(ytest.eval() == cls))

    # final ids
    final_ids = numpy.copy(train_ids)
    nitems    = len(final_ids)/2
    final_ids.resize((2,nitems))
    trainfinal_ids = final_ids[0]
    valfinal_ids   = final_ids[1]

    xtrain,ytrain,minv,maxv = get_data( dataset, trainfinal_ids, options )
    xval,yval    = get_data( dataset, valfinal_ids, options, isFirst = False, minvalue=minv, maxvalue=maxv )[0:2]
    xtest,ytest  = get_data( dataset, test_ids    , options, isFirst = True , minvalue=minv, maxvalue=maxv )[0:2]

    trainFinal = (xtrain,ytrain)
    valFinal   = (xval,yval)
    testFinal  = (xtest,ytest)

    print >> sys.stderr, test_ids
    
    if options['verbose'] > 0:
        print 'Train set with size %d ' % (ytrain.shape.eval())
        print 'Val  set with size %d ' % (yval.shape.eval())
        print 'Test  set with size %d ' % (ytest.shape.eval())
        if options['verbose'] > 5:
            for cls in range(0,2):
                print >>sys.stderr, "\tNumber of training elements for cls {0:02d} is {1:05d}".format(cls,sum(ytrain.eval() == cls))
                print >>sys.stderr, "\tNumber of validation elements for cls {0:02d} is {1:05d}".format(cls,sum(yval.eval() == cls))
                print >>sys.stderr, "\tNumber of testing elements for cls {0:02d} is {1:05d}".format(cls,sum(ytest.eval() == cls))

    basefilename = '{0:s}/{1:05d}_{2:03d}_'.format(options['outputfolder'],nrun,string.atoi(options['resolution']))

    trainfilename = basefilename + 'train_ids.pkl.gz'
    valfilename   = basefilename + 'val_ids.pkl.gz'
    trainfinalfilename = basefilename + 'trainfinal_ids.pkl.gz'
    valfinalfilename   = basefilename + 'valfinal_ids.pkl.gz'
    testfilename       = basefilename + 'test_ids.pkl.gz'

    save_gzdata(trainfilename,train_ids)
    save_gzdata(valfilename,val_ids)
    save_gzdata(trainfinalfilename,trainfinal_ids)
    save_gzdata(valfinalfilename,valfinal_ids)
    save_gzdata(testfilename,test_ids)
    
    if options['verbose'] > 0:
        print 'Train set with size %d' % (trainFinal[1].shape.eval())
        print 'Val   set with size %d' % (valFinal[1].shape.eval())
        print 'Test  set with size %d' % (testFinal[1].shape.eval())

    rval = [trainval, valval, testval, trainFinal, valFinal, testFinal ]

    return rval

#train_set, valid_set, test_set format: tuple(input, target)
#input is an numpy.ndarray of 2 dimensions (a matrix)
#witch row's correspond to an example. target is a
#numpy.ndarray of 1 dimensions (vector)) that have the same length as
#the number of rows in the input. It should give the target
#target to the example with the same index in the input.

def shared_dataset(data_x,data_y, borrow=True):
    """ Function that loads the dataset into shared variables
    
    The reason we store our dataset in shared variables is to allow
    Theano to copy it into the GPU memory (when code is run on GPU).
    Since copying data into the GPU is slow, copying a minibatch everytime
    is needed (the default behaviour if the data is not in a shared
    variable) would lead to a large decrease in performance.
    """
    shared_x = theano.shared(numpy.asarray(data_x,
                                           dtype=theano.config.floatX),
                             borrow=borrow)
    shared_y = theano.shared(numpy.asarray(data_y,
                                           dtype=theano.config.floatX),
                             borrow=borrow)
    # When storing data on the GPU it has to be stored as floats
    # therefore we will store the labels as ``floatX`` as well
    # (``shared_y`` does exactly that). But during our computations
    # we need them as ints (we use labels as index, and if they are
    # floats it doesn't make sense) therefore instead of returning
    # ``shared_y`` we will have to cast it to int. This little hack
    # lets ous get around this issue
    return shared_x, T.cast(shared_y, 'int32')

