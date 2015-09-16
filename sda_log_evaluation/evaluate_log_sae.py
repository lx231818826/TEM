#!/usr/bin/python
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
import sys, string, os, copy, time, re, csv
import itertools, numpy, h5py
# opencv
import cv2

lib_path = os.path.abspath('../TL/')
sys.path.append(lib_path)

from SdA import *

from data_preprocessing import load_data, gen_folds, confusion_matrix, shared_dataset
from data_handling import save_results, save_data, load_saveddata,load_savedgzdata, save_gzdata
from matplotlib import pyplot as plt

# 
def print_usage():
    print './main.py resolution method'
    print """

resolution:
\t - all
\t - 15000
\t - 20000
\t - 30000
\t - 50000

method:
\t - baseline
\t - tl
"""

def checkResults(nelem_x, nmbrAnn, anncenters, pt, ypred, mindistgiven=10):
    # --------------------------------------------------------------------------------
    # check results
    TP = 0; FP = 0; FN = 0;
    
    GT  = numpy.full((nmbrAnn,),0,dtype=numpy.uint8)

    for i in range(0,nelem_x):
        # background
        if ypred[i] == 1:
            continue
        
        pti = pt[:,i]
        pti = pti.reshape((2,-1))
        pti = numpy.tile(pti,(1,nmbrAnn))

        # calculates the distance of one detection against all anotations
        dists      = numpy.sqrt( numpy.sum( (pti - anncenters) * (pti - anncenters), axis=0) )
        mindistidx = numpy.nonzero( dists <= mindistgiven )[0]
        
        nanofound = False
        if len( mindistidx ) > 0:
            mindistance       = dists[mindistidx]
            orderedmindistidx = numpy.argsort( mindistance )

            for arg in orderedmindistidx:
                # print arg,
                # raw_input()
                
                if GT[mindistidx[arg]] == 0:
                    GT[mindistidx[arg]] = 1
                    TP = TP + 1
                    nanofound = True
                    break
                
            if nanofound == False:
                FP = FP + 1
                
        else:
            FP = FP + 1

    FN = sum(GT == 0)

    # print FN + TP, nmbrAnn
    # print TP
    return (TP,FP,FN)

# ---------------------------------------------------------------------------------------------------------------------
def getPrecisionRecall(nfiles,files,ids,path,imgsbasepath,imgspath,annbasepath,annfiles, model, (rd,th,nrunImg,cv), printImg=False):
    resize   = 1.
    minvalue = 0.
    maxvalue = 255.
        
    # --------------------------------------------
    Precision = 0
    Recall    = 0

    Precision_LoG = 0
    Recall_LoG    = 0
    # --------------------------------------------
    nDetections = 0

    count = -1
    for n in range(0,nfiles,2):
        count = count + 1
        filepathx = path + files[n]
        filepathy = path + files[n+1]

        #print >> sys.stderr, filepathx
        #print >> sys.stderr, filepathy
        
        # get x,y of LoG detections
        fx = h5py.File(filepathx,'r')
        fy = h5py.File(filepathy,'r')

        detectedx = fx.get('data')
        #print numpy.array( detectedx )
        detectedx = numpy.array( detectedx, dtype=numpy.float ) # samples were resized
        detectedy = fy.get('data')
        detectedy = numpy.array( detectedy, dtype=numpy.float ) # samples were resized
        
        # get imgs
        print >> sys.stderr, "loading... {0:s}".format( imgspath[ids[count]] )
        img = cv2.imread(imgsbasepath + imgspath[ids[count]])
        height, width, depth = img.shape
        height = height / resize
        width  = width / resize
        # print "heigth {0:02d} width {1:03d}".format(height,width)
        
        # get annotation
        print >> sys.stderr, "loading..: {0:s}".format( annfiles[ids[count]] )
        # print >> sys.stderr, annfiles[ids[count]]
        csvfile = open(annbasepath + annfiles[ids[count]],'r')
        csvreader = csv.reader(csvfile, delimiter=',')
        next(csvreader, None) # skip header
        countann = 0
        for line in csvreader:
            xtopleft     = int(float(line[0])) # to avoid precision errors and to be equivalent to the matlab implementation
            xbottomright = int(float(line[2]))
            ytopleft     = int(float(line[1]))
            ybottomright = int(float(line[3]))
            
            # top-left point, bottom-right point
            xc = int( (xtopleft + xbottomright) /2 / resize + .5)
            yc = int( (ytopleft + ybottomright) /2 / resize + .5)
            
            centers = [xc, yc]
            
            if countann == 0:
                anncenters = centers
            else:
                anncenters = numpy.c_[anncenters, centers]
            countann = countann + 1

        nmbrAnn   = anncenters.shape[1]
        toDiscard = numpy.zeros((len(detectedx),1))

        # print nmbrAnn
        # alla
        isFirst = True
        for i in range(0,len(detectedx)):

            if detectedy[i]-10 < 0 or detectedy[i]+10 > height or \
               detectedx[i]-10 < 0 or detectedx[i]+10 > width:
                toDiscard[i] = 1
                continue

            d = img[detectedy[i]*resize-10:detectedy[i]*resize+10, detectedx[i]*resize-10:detectedx[i]*resize+10,1]
            
            ndim = numpy.size(d)
            d = numpy.reshape(d, (-1,ndim))
            d = d.astype(dtype=numpy.float)
            
            # print [detectedx[i], detectedy[i]]
            if isFirst == True:
                # valid points
                pt = numpy.c_[detectedx[i], detectedy[i]].T

                # patch
                data = numpy.c_[d].T
                isFirst = False
            else:
                p  = numpy.c_[detectedx[i], detectedy[i]].T
                pt = numpy.c_[pt, p]

                d = numpy.c_[d].T
                data = numpy.c_[data, d]

        set_x   = data[0:,].T
        nelem_x = set_x.shape[0]

        #minvalue = numpy.min(set_x)
        #maxvalue = numpy.max(set_x)
        
        set_x = (set_x - minvalue) / (maxvalue-minvalue+0.001)
        set_y = numpy.zeros((nelem_x,),dtype=numpy.float)

        ypredlog = numpy.zeros((nelem_x,)) # all detections
        ( TP, FP, FN ) = checkResults( nelem_x, nmbrAnn, anncenters, pt, ypredlog, mindistgiven=4/resize )
        print >> sys.stderr, "(LoG) TP: {0:05d} | FP: {1:05d} | FN: {2:05d} | N: {3:05d}".format(TP, FP, FN, nmbrAnn)
        Precision_LoG_ = TP/(TP+FP+0.0001)
        Recall_LoG_    = TP/(TP+FN+0.0001)

        test_data_x, test_labels_y = shared_dataset(set_x,set_y)
        
        test_model = model.build_test_function(
            dataset       = (test_data_x,test_labels_y),
            batch_size    = 1,
        )

        (ytrue,ypred,yprob) = test_model()
        #ypred = numpy.array( map(lambda x: not x>.6,numpy.amax(yprob,axis=1)), dtype=numpy.uint8)
        # ypred = numpy.zeros((nelem_x,)) # all detections
        # print 'No samples: {0:03d}'.format(len(ytrue))
        
        (TP,FP,FN) = checkResults(nelem_x, nmbrAnn, anncenters, pt, ypred, mindistgiven=20/resize)[0:3]
        print >> sys.stderr, "(SdA) TP: {0:05d} | FP: {1:05d} | FN: {2:05d} | N: {3:05d}".format(TP, FP, FN, nmbrAnn)
        Precision_ = TP/(TP+FP+0.0001)
        Recall_    = TP/(TP+FN+0.0001)

        Precision_LoG = Precision_LoG + Precision_LoG_
        Recall_LoG    = Recall_LoG + Recall_LoG_
        
        Precision  = Precision + Precision_
        Recall     = Recall    + Recall_


        nDetections = nDetections + len(detectedx)

        if printImg:
            img = cv2.resize(img,(0,0),fx=1/resize,fy=1/resize)

            for i in range(0,nmbrAnn):
                ctr = anncenters[:,i]
                # green, annotation
                cv2.circle(img,(int(ctr[0]),int(ctr[1])),10,(0,255,0),5)
                
            for i in range(0,nelem_x):
                pti = pt[:,i]

                if ypredlog[i] == 0:
                    # blue, log
                    cv2.circle(img,(int(pti[0]),int(pti[1])),20,(255,0,0),5)

                if ypred[i] == 0:
                    # red, SdA
                    cv2.circle(img,(int(pti[0]),int(pti[1])),30,(0,0,255),5)

            filename = "imgs_debug/{0:s}_r={1:d}_th={2:d}_{3:03d}_cv={4:d}.jpg".format(imgspath[ids[count]],rd,th,nrunImg,cv)
            print >> sys.stderr, "Saving image..:" + filename
            cv2.imwrite(filename,img)

            filename = "imgs_debug/{0:s}_r={1:d}_th={2:d}_{3:03d}_cv={4:d}_LoG.pkl.gz".format(imgspath[ids[count]],rd,th,nrunImg,cv)
            print >> sys.stderr, "(LoG) Precision: {0:05f} | Recall: {1:05f} ".format(Precision_LoG_, Recall_LoG_)
            save_gzdata(filename,[Precision_LoG_,Recall_LoG_])

            filename = "imgs_debug/{0:s}_r={1:d}_th={2:d}_{3:03d}_cv={4:d}.pkl.gz".format(imgspath[ids[count]],rd,th,nrunImg,cv)
            print >> sys.stderr, "(SdA) Precision: {0:05f} | Recall: {1:05f} ".format(Precision_, Recall_)
            save_gzdata(filename,[Precision_,Recall_])

            print >> sys.stderr, ("Ann: {0:05d} | Nano (SdA): {1:05d}| Back (SdA): {2:05d}| LoG: {3:05d} ").format(nmbrAnn, sum(numpy.array(ypred)==0), sum(numpy.array(ypred)==1), nelem_x)
            print >> sys.stderr, "-------------------------"

            
    # average over all files
    Precision = Precision / (nfiles/2)
    Recall    = Recall / (nfiles/2)

    Precision_LoG = Precision_LoG / (nfiles/2)
    Recall_LoG    = Recall_LoG / (nfiles/2)

    nDetections   = nDetections / (nfiles/2.)
    print >> sys.stderr, ("number of detections: {0:05f}").format(nDetections)
    return (Precision, Recall, Precision_LoG, Recall_LoG, nDetections)

# -------------------------------------------------------------------------------------
def main(resolution,method,pathRes):
    # load results from LoG
    
    imgpathsae  = '../../imgs_nanoparticles/{0:03d}/db2/resultado_sae/'.format(string.atoi(resolution))

    if method == 'baseline':
        basepath = './{0:s}/{1:05d}/models/res_baseline_resized_{1:05d}_111111/'.format(pathRes,string.atoi(resolution))
    elif method == 'tl':
        basepath = './{0:s}/{1:05d}/models/res_tl_resized_50000_{1:05d}_111111/'.format(pathRes,string.atoi(resolution))
    
    # annotations
    annbasepath = '../../imgs_nanoparticles/{0:03d}/db2/annotation/user/'.format(string.atoi(resolution))
    annfiles = [f for f in os.listdir(annbasepath) if re.match(r'[\w\W]*csv', f)]
    annfiles = sorted( annfiles )

    # imgs base paths
    imgsbasepath = '../../imgs_nanoparticles/{0:03d}/db2/'.format(string.atoi(resolution))
    imgspath = os.listdir(imgsbasepath)
    imgspath = sorted( imgspath )

    # ------------------------------------------------------------------------------------------------
    # TEST DATA

    PrecisionAll = []
    RecallAll    = []

    PrecisionLoGAll = []
    RecallLoGAll    = []

    nDetectionsAll  = []
    
    for nrun in range(1,21): #
        print >> sys.stderr, "\n**************************\n"
        print >> sys.stderr, "NRUN {0:05d}/20 ".format(nrun)
        
        filename = '{0:s}/{1:05d}_{2:03d}_model.pkl.gz'.format(basepath,nrun,string.atoi(resolution))
        print >> sys.stderr, "Loading " + filename
        model    = load_savedgzdata(filename)

        # get ids
        pathids = '{0:s}/{1:05d}_{2:05d}_test_ids.pkl.gz'.format(basepath,nrun,string.atoi(resolution))
        print >> sys.stderr, 'Loading ' + pathids + '...'
        ids = load_savedgzdata(pathids)
        print >> sys.stderr, ids
        
        reg = 'detectedNanoParticlesDetectionResult_log_detector_test_{0:03d}_'.format(nrun)
        files = [f for f in os.listdir(imgpathsae) if re.match(reg, f)]
        # order data
        files = sorted( files )
        
        nfiles = len(files)

        (Precision, Recall, PrecisionLoG,RecallLoG,nDetections) = getPrecisionRecall(nfiles,files,ids,imgpathsae,imgsbasepath,imgspath,annbasepath,annfiles,model,(0,0,nrun,0),printImg=True)
        
        print >> sys.stderr, "Precision LoG: {0:05f} | Recall LoG: {1:05f}".format(PrecisionLoG, RecallLoG)
        print >> sys.stderr, "Precision SdA: {0:05f} | Recall SdA: {1:05f}".format(Precision, Recall)
        # kaka
        
        PrecisionAll.append( Precision )
        RecallAll.append( Recall )

        PrecisionLoGAll.append( PrecisionLoG )
        RecallLoGAll.append( RecallLoG )

        nDetectionsAll.append( nDetections )
        
    # ---------------------------------------------------------
    PrecisionAll = numpy.array( PrecisionAll ) 
    RecallAll    = numpy.array( RecallAll )

    PrecisionLoGAll = numpy.array( PrecisionLoGAll )
    RecallLoGAll    = numpy.array( RecallLoGAll )

    nDetectionsAll  = numpy.array( nDetectionsAll )
    
    print "--------------------------------------------\n"
    print "Precision LoG: {0:03f} ({1:03f}) | Recall LoG: {2:03f} ({3:03f})".format(numpy.mean(PrecisionLoGAll),numpy.std(PrecisionLoGAll),numpy.mean(RecallLoGAll),numpy.std(RecallLoGAll))
    print "Precision SdA: {0:03f} ({1:03f}) | Recall SdA: {2:03f} ({3:03f})".format(numpy.mean(PrecisionAll),numpy.std(PrecisionAll),numpy.mean(RecallAll),numpy.std(RecallAll))
    print "number detections: {0:03f} ({1:03f})".format(numpy.mean(nDetectionsAll),numpy.std(nDetectionsAll))
    
    PrecisionRecall = numpy.c_[PrecisionAll,RecallAll]
    filename = 'results/sae_{0:s}_{1:s}_test_all.pkl.gz'.format(method,resolution)
    save_gzdata(filename, PrecisionRecall )

    PrecisionRecallLoG = numpy.c_[PrecisionLoGAll,RecallLoGAll]
    filename = 'results/log_{0:s}_{1:s}_test_all.pkl.gz'.format(method,resolution)
    save_gzdata(filename, PrecisionRecallLoG )

    PrecisionRecall = numpy.r_[numpy.mean(PrecisionAll),numpy.mean(RecallAll)]
    filename = 'results/sae_{0:s}_{1:s}_test.pkl.gz'.format(method,resolution)
    save_gzdata(filename, PrecisionRecall )

    PrecisionRecallLoG = numpy.r_[numpy.mean(PrecisionLoGAll),numpy.mean(RecallLoGAll)]
    filename = 'results/log_{0:s}_{1:s}_test.pkl.gz'.format(method,resolution)
    save_gzdata(filename, PrecisionRecallLoG )

    filename = 'results/ndetections_{0:s}_{1:s}_test.pkl.gz'.format(method,resolution)
    save_gzdata(filename, nDetectionsAll )

        
# -------------------------------------------------------------------------------------
if __name__ == '__main__':
    print os.sys.argv
    if not( len(os.sys.argv) == 4 ):
        print_usage()
        os.sys.exit(-1)

    print os.sys.argv

    # execute experiment
    main(os.sys.argv[1],os.sys.argv[2],os.sys.argv[3])
