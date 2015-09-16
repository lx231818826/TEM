#!/usr/bin/python
# -*- coding: iso-8859-15 -*-

import gzip, sys, os, re, csv, string, h5py
from myplotFunctions import *
from numpy import *
import cPickle as pickle

# author: Ricardo Sousa
# rsousa at rsousa.org

# webcolourdata.com
# http://lokeshdhakar.com/projects/color-thief/
# http://designsparkle.com/color-palette-generators/

project_colors = ['#74A82A', '#D55511', '#3565A1', '#58616A', '#A42F11', '#DC6D1C', '#C94612' , '#E67217', '#EEA435', '#6BA2D0', '#B7C4CF', '#75D0ED'];
project_markers   = ['d','o','v','<','>','^','d','o','v','<','>','^']
project_linestyle = ['-','-.',':']
project_edgecolor = ['#ffffff']

styles = {'pcolors':project_colors, 'pmarkers': project_markers, \
            'plinestyle': project_linestyle, 'pedgecolor': project_edgecolor, \
            'fontsize': 40}

legendsTitle = {'15000': '(a)', \
                '20000': '(b)', \
                '30000': '(c)', \
                '50000': '(d)', \
            }

# radius legends
logLegendsRadius = {'15000':array(xrange(3,6)), \
                    '20000':array([3, 5, 7, 9]), \
                    '30000':array([5, 7, 9, 11]), \
                    '50000':array([9, 11, 13])}

logLegendsRadiusStep = {'15000': min(logLegendsRadius['15000'])-3, \
                        '20000': min(logLegendsRadius['20000'])-3, \
                        '30000': min(logLegendsRadius['30000'])-3, \
                        '50000': min(logLegendsRadius['50000'])-3}

name = {
    '15000': 'db1',
    '20000': 'db2',
    '30000': 'db3',
    '50000': 'db4'
}

#scale      = {[3 7], [7 13], [7 13]};
#size       = {[0 10 15], [0 10 15], [0 10 15]};

def usage(name):
    print name + " method dataset resolution [withcv]"
    print """
method:
\t - icy
\t - log_detector

dataset:
\t - db1
\t - db2

resolution:
\t - all
\t - 15000
\t - 20000
\t - 30000
\t - 50000

"""

# check arguments
def parse(arguments):
    wrongMethod     = arguments['method'] != 'icy' and arguments['method'] != 'log_detector'
    wrongDataset    = arguments['dataset'] != 'db1' and arguments['dataset'] != 'db2'
    wrongResolution = arguments['resolution'] != 'all' and arguments['resolution'] != '15000' and \
        arguments['resolution'] != '20000' and arguments['resolution'] != '30000' and \
        arguments['resolution'] != '50000' 
    return wrongMethod or wrongDataset or wrongResolution


def loadMatFiles(filename):
    
    for nrun in range(1,21):
        filename_ = '{0:s}_{1:03d}.mat'.format(filename,nrun)
        
        print 'Loading file.. ' + filename_
        try:
            f = h5py.File(filename_,'r')
        except Exception, e:
            raise e
        
        ROC_precision_    = f.get('ROC_precision')
        ROC_recall_       = f.get('ROC_recall')
        ROC_varPrecision_ = f.get('ROC_varPrec')
        ROC_varRecall_    = f.get('ROC_varReca')

        if nrun == 1:
            ROC_precision    = array(ROC_precision_)
            ROC_recall       = array(ROC_recall_)
            ROC_varPrecision = array(ROC_varPrecision_)
            ROC_varRecall    = array(ROC_varRecall_)

        else:
            # continue
            # print ROC_precision

            ROC_precision    = ROC_precision    + array(ROC_precision_)
            ROC_recall       = ROC_recall       + array(ROC_recall_)
            ROC_varPrecision = ROC_varPrecision + array(ROC_varPrecision_)
            ROC_varRecall    = ROC_varRecall    + array(ROC_varRecall_)

    ROC_precision = ROC_precision / 20
    ROC_recall    = ROC_recall / 20
        
    return (ROC_precision,ROC_recall)


def doPlots(ROC_precision,ROC_recall,method,resolution,dataset,withcv, \
            figureref = None, axisref = None, leg = None, counter = 0, lim = None,
            last=False):

    if method == 'icy':
        (fig,ax) = myPlotPrecRecICY(ROC_precision,ROC_recall,styles,\
                                    legendsTitle[resolution] + ' SD' + ' ' + resolution,\
                                    icyLegendsSize[resolution], icyLegendsScale[resolution], dataset, withcv, figureref, axisref)
    elif method == 'log_detector':
        # + ' + SdA '
        (fig,ax,count, limits) = myPlotPrecRecLOGDetector(ROC_precision,ROC_recall,styles,\
                                                          legendsTitle[resolution] + ' ' +  'Detector ' + name[resolution],\
                                                          logLegendsRadius[resolution], logLegendsRadiusStep[resolution], \
                                                          dataset, withcv, figureref, axisref, \
                                                          leg, counter, lim,last)

    return (fig,ax,count, limits)

# ------------------------------------------------------------------------------------------
def loadResultsFile(filename,method,resolution,dataset,withcv=False):

    print withcv

    if not withcv:
        filename = '{0:s}/resultado_sae/GlobalNanoParticlesDetectionResult_{1:s}.mat'.format(filename,method)
    else:
        
        val = '{0:s}/resultado_sae/GlobalNanoParticlesDetectionResult_{1:s}_val'.format(filename,method)
        test  = '{0:s}/resultado_sae/GlobalNanoParticlesDetectionResult_{1:s}_test'.format(filename,method)

        filenameRes = {'val': val, 'test': test}


    if type(filenameRes) is dict:
        ( ROC_precision, ROC_recall ) = loadMatFiles(filenameRes['val'])
        (fig, ax, counter, limits) = doPlots(ROC_precision,ROC_recall,method,resolution,dataset,False)

        ( ROC_precision, ROC_recall ) = loadMatFiles(filenameRes['test'])
        (fig, ax, counter, limits) = doPlots(ROC_precision,ROC_recall,method,resolution,dataset,withcv,fig,ax, counter=counter,leg="Test: LoG", lim=limits)
        print ROC_precision
        print ROC_recall
        
        data = pickle.load(gzip.open( "../results/sae_baseline_" + resolution + "_test.pkl.gz", "rb" ) )
        ROC_precision = full((1,1),data[0],dtype=float)
        ROC_recall    = full((1,1),data[1],dtype=float)
        (fig, ax, counter, limits) = doPlots(ROC_precision,ROC_recall,method,resolution,dataset,withcv,fig,ax,leg="Test: LoG+SdA",counter=counter,lim=limits,last=True)
        print ROC_precision
        print ROC_recall
        
    else:
        ( ROC_precision, ROC_recall ) = loadMatFiles(filename)
        doPlots(ROC_precision,ROC_recall,method,resolution,dataset,False)
        
    
# ------------------------------------------------------------------------------------------
def loadResults(arguments):

    basepath = '../../../imgs_nanoparticles'
    if arguments['resolution'] != 'all':
        path = '{0:s}/{1:s}/{2:s}'.format(basepath,arguments['resolution'],arguments['dataset'])
        #print path

        resolution = arguments['resolution']
        
        loadResultsFile(path, arguments['method'],resolution,arguments['dataset'],arguments['withcv'])
    else:
        resolutions = ['15000','20000','30000','50000']
        for resolution in resolutions:
            path = '{0:s}/{1:s}/{2:s}'.format(basepath,resolution,arguments['dataset'])

            loadResultsFile(path, arguments['method'],resolution,arguments['dataset'],arguments['withcv']) 

# -----------------------------------------------------------------------------------------------------
# main function
def main(argv):
    if len(argv) < 4:
        usage(argv[0])
        return -1
    elif len(argv) < 5 :
        withcv = False
    elif argv[4] == 'withcv':
        withcv = True
    else:
        usage(argv[0])
        return -1

    arguments = {'method':argv[1],'dataset':argv[2],'resolution':argv[3],'withcv':withcv}
    if parse(arguments):
        usage(argv[0])
        return -1

    print 'Generating the plots..'

    print arguments

    #print 'Plot Init..'
    loadResults(arguments)
    
    return

if __name__ == "__main__":
    sys.exit( main(sys.argv) )
