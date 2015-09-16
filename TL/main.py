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

## ---------------------------------------------------------------------------
import time, csv, copy, os, sys, string

# DEBUG INFORMATION
print >> sys.stderr, "\n\n ------------- CHECK GPU NUMBER --------------------------\n\n"

import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

from mlp import HiddenLayer
from dA import dA
from SdA import SdA

# --------------------------------------------------------------------------------------------------------------- 
import itertools, numpy
numpy.set_printoptions(threshold=numpy.nan)

from theano.tensor.shared_randomstreams import RandomStreams

from data_preprocessing import load_data, gen_folds, confusion_matrix

from data_handling import save_results, save_gzdata, load_savedgzdata


# DEBUG INFORMATION
print >> sys.stderr, "\n\n ------------- CHECK GPU NUMBER --------------------------\n\n"



# --------------------------------------------------------------------------------------------------------------- 
def print_usage():
    print './main.py datasetpath [retrain_ft_layers]'
 
def evaluate_error(y_valid, y_pred, options ):
    # F-Score
    # label 0 = nano
    # label 1 = back

    background   = 1
    nanoparticle = 0
    
    TP = 0
    FN = 0
    FP = 0
    TN = 0
    
    for ith in range(0,len(y_pred)):
        if y_pred[ith]   == background   and y_valid[ith] == background:
            TN = TN + 1
        elif y_pred[ith] == nanoparticle and y_valid[ith] == background:
            FP = FP + 1
        elif y_pred[ith] == background   and y_valid[ith] == nanoparticle:
            FN = FN + 1
        elif y_pred[ith] == nanoparticle and y_valid[ith] == nanoparticle:
            TP = TP + 1
            
    Recall    = TP / (TP+FN+0.0001)
    Precision = TP / (TP+FP+0.0001)

    if options['measure'] == 'fmeasure':
        test_scores = 1 - 2*(Precision*Recall) / (Precision+Recall+0.0001)
    elif options['measure'] == 'f1score':
        test_scores = 1 - 2*TP / (2*TP + FP + FN + 0.0001)
    elif options['measure'] == 'acc':
        test_scores = 1 - (TP + TN) / ( TP + TN + FP + FN + 0.0001 )
    elif options['measure'] == 'weightedmer':
        weights = numpy.zeros( y_valid.shape, dtype=numpy.float32 )
        ind = y_valid == background   # background

        weights[ ind] = options['weight']
        weights[~ind] = 1
        
        test_scores = weights * numpy.array(y_valid != y_pred, dtype=numpy.float)
        test_scores = numpy.mean( test_scores )

    elif options['measure'] == 'weightedmercls':
        indback = y_valid == background
        indnano = y_valid == nanoparticle
        ninstback = sum(indback)
        ninstnano = sum(indnano)

        weightscls = numpy.zeros( y_valid.shape, dtype=numpy.float32 )
        # print >> sys.stderr, "back: {0:f} | nano: {1:f}".format(ninstback,ninstnano)
        val = numpy.minimum(ninstback,ninstnano)/(numpy.maximum(ninstback,ninstnano)*1.)
        # print >> sys.stderr, val
        
        if ninstback > ninstnano:
            weightscls[indnano] = 2-val
            weightscls[indback] = 1
        else:
            weightscls[indnano] = 1
            weightscls[indback] = 2-val
        
        weights = numpy.zeros( y_valid.shape, dtype=numpy.float32 )
        ind = y_valid == background   # background

        weights[ ind] = options['weight']
        weights[~ind] = 1
        
        test_scores = weightscls  * numpy.array(y_valid != y_pred, dtype=numpy.float)
        test_scores = numpy.mean( test_scores )
    
    #print >> sys.stderr, test_scores
    
    return test_scores

    
# -------------------------------------------------------------------------------------
def evaluate_model(sda,testdata,options):

    train_set_x, train_set_y = testdata
    
    if options['retrain'] == 0:    
        test_model = sda.build_test_function(
            dataset       = testdata,
            batch_size    = 1, # options['batchsize']
            )
    else:
        test_model = sda.build_test_function_reuse(
            dataset       = testdata,
            batch_size    = 1, #options['batchsize'],
        )

    # print sda.params[1].get_value()[-1]
    # print sda.params_b[1].get_value()[-1]
    # kkk

    (ytest,ypred,ypred_prob) = test_model()

    print >> sys.stderr, "Test GT Differences: "
    print >> sys.stderr, sum(ytest != train_set_y.eval())

    if options['oneclass'] == True:
        options['nclasses'] = 2
    
    # ypred      = numpy.array( ypred_prob[:,0] < options['threshold'], dtype=numpy.uint8)
    if options['threshold'] != None:
        ypred = numpy.array( ypred_prob[:,0] < options['threshold'], dtype=numpy.uint8)
    else:
        ypred = numpy.argmax( ypred_prob, axis = 1 )

    
    test_score = evaluate_error( ytest, ypred, options )

    cm = confusion_matrix( ytest, ypred, options['nclasses'] )

    print >> sys.stderr, "Test CM"
    print >> sys.stderr, cm
    
    if options['verbose'] > 0:
        print >> sys.stderr, (('     test error of best model %03f %%') %
                              (test_score * 100.))

    return (test_score, ytest, ypred)
    
# -------------------------------------------------------------------------------------
def pretrain_finetune_model(sda,pretraining_fns,train_set,test_set,options):
    train_set_x, train_set_y = train_set

    n_train_batches  = train_set_x.get_value(borrow=True).shape[0]
    n_train_batches /= options['batchsize']

    if options['retrain'] == 0:    
    
        bestmodelsda = copy.copy( sda )
        # -----------------------------------------------
        # PRETRAINING
        # -----------------------------------------------  
        if options['verbose'] > 5:
            print >> sys.stderr, ('... pre-training the model')
        start_time = time.clock()
        ## Pre-train layer-wise
        corruption_levels = options['corruptlevels']
        for i in xrange(sda.n_layers):
            # go through pretraining epochs
            for epoch in xrange(options['pretraining_epochs']):
                # go through the training set
                c = []
                for batch_index in xrange(n_train_batches):
                    c.append(pretraining_fns[i](index=batch_index,
                                                corruption=corruption_levels[i],
                                                lr=options['pretrain_lr']))

                if epoch % 100 == 0 and options['verbose'] > 5:
                    print >> sys.stderr, ('Pre-training layer %02i, epoch %04d, cost ' % (i, epoch)),
                    print >> sys.stderr, (numpy.mean(c))
        end_time = time.clock()
        if options['savetimes']:
            filename = '{0:s}/times_pr_{1:03d}_{2:03d}.pkl.gz'.format(options['outputfolderres'],options['nrun'],string.atoi(options['resolution']))
            save_gzdata(filename, end_time - start_time)
        
        if options['verbose'] > 4:
            print  >> sys.stderr, ('The pretraining code for file ' +
                                   os.path.split(__file__)[1] +
                                   ' ran for %.2fm' % ((end_time - start_time) / 60.))

        # get the training, validation and testing function for the model
        #dataset = [folds[0][0], folds[1][0], folds[2]]
        dataset = [train_set, test_set]
    
        if options['verbose'] > 5:
            print >> sys.stderr,('... getting the finetuning functions')
        train_fn, validate_model = sda.build_finetune_functions(
            datasets=dataset,
            batch_size=options['batchsize'],
            learning_rate=options['finetune_lr']
        )

    else:
        dataset = [train_set, test_set]

        train_fn, validate_model = sda.build_finetune_functions_reuse(
            datasets=dataset, batch_size=options['batchsize'],
            learning_rate=options['finetune_lr'], update_layerwise=options['retrain_ft_layers'])
        
    # ------------------------------------------------------------------------------------------------
        
    # -----------------------------------------------
    # FINETUNE
    # -----------------------------------------------  
    if options['verbose'] > 5:
        print >> sys.stderr, ('... finetunning the model')
    # early-stopping parameters
    patience = 10 * n_train_batches  # look as this many examples regardless
    patience_increase = 2. #2. # wait this much longer when a new best is found
    improvement_threshold = 0.995 # 0.995  # a relative improvement of this much is
                                   # considered significant
    validation_frequency = min(n_train_batches, patience / 2)
                                  # go through this many
                                  # minibatche before checking the network
                                  # on the validation set; in this case we
                                  # check every epoch

    best_validation_loss = numpy.inf
    test_score = 0.

    start_time = time.clock()

    done_looping = False
    epoch = 0

    while (epoch < options['training_epochs']) and (not done_looping):
        epoch = epoch + 1
        for minibatch_index in xrange(n_train_batches):
            minibatch_avg_cost = train_fn(minibatch_index)

            iter = (epoch - 1) * n_train_batches + minibatch_index

            if (iter + 1) % validation_frequency == 0:
                #this_validation_loss    = numpy.mean( validate_model() )
                (y_valid, y_pred, y_pred_prob)  = validate_model()

                # pos = numpy.random.randint(len(y_pred),size=(100,))
                # print options
                # print pos
                # print y_pred_prob[pos,:].T
                # raw_input()
                # alll

                # we are going to control the predictions according to their prob
                if options['threshold'] != None:
                    y_pred = numpy.array( y_pred_prob[:,0] < options['threshold'], dtype=numpy.uint8)
                else:
                    y_pred = numpy.argmax( y_pred_prob, axis = 1 )

                this_validation_loss = evaluate_error( y_valid, y_pred, options )

                # if epoch % 10 == 0:
                #     cm = confusion_matrix(y_valid, y_pred, options['nclasses'])
                #     print >> sys.stderr, cm, this_validation_loss
                    
                # print >> sys.stderr, this_validation_loss
                # this_validation_loss = numpy.mean(validation_losses)
                if  epoch % 30 == 0 and options['verbose'] > 5:
                    # print >> sys.stderr, y_valid
                    # print >> sys.stderr, y_pred_prob
                    # print >> sys.stderr, y_pred
                    # print >> sys.stderr, y_valid.shape
                    # print >> sys.stderr, y_pred.shape
                    # print >> sys.stderr, y_pred_prob[1:10,:], y_pred[1:10], y_valid[1:10]
                    # print >> sys.stderr, test_set[1].eval()
                    
                    print >> sys.stderr,('epoch %04i, minibatch %04i/%04i, validation error %03f %%' %
                                         (epoch, minibatch_index + 1, n_train_batches,
                                          this_validation_loss * 100.))

                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:
                    bestmodelsda = copy.copy(sda)

                    # % ------------------------------------------------------------
                    if options['oneclass'] == True:
                        options['nclasses'] = 2

                    #print >> sys.stderr, sda.params[-2].get_value().T, sda.params[-1].get_value()
                    pos = numpy.random.randint(len(y_pred),size=(10,))
                    # print options
                    # print pos
                    # print y_pred_prob.shape
                    #print >> sys.stderr, options['threshold']
                    # print >> sys.stderr, numpy.array( y_pred_prob[:,0] < options['threshold'], dtype=numpy.uint8)
                    #print >> sys.stderr, y_pred_prob[pos,:].T
                    #print >> sys.stderr, y_pred[pos]
                    cm = confusion_matrix(y_valid, y_pred, options['nclasses'])
                    #print >> sys.stderr, ("Fine tune...epoch %04i" %  epoch)
                    #print >> sys.stderr, this_validation_loss
                    #print >> sys.stderr, cm
                    # options['nclasses'] = 1
                    # % ------------------------------------------------------------

                    
                    # improve patience if loss improvement is good enough
                    if (
                            this_validation_loss < best_validation_loss *
                            improvement_threshold
                    ):
                        patience = max(patience, iter * patience_increase)

                    # save best validation score and iteration number
                    best_validation_loss = this_validation_loss
                    best_iter = iter

                    if patience <= iter:
                        done_looping = True
                        break

    end_time = time.clock()

    if options['savetimes']:
        filename = '{0:s}/times_fn_{1:03d}_{2:03d}.pkl.gz'.format(options['outputfolderres'],options['nrun'],string.atoi(options['resolution']))
        save_gzdata(filename, end_time - start_time)

    print >> sys.stderr, ("Stopped at epoch %04i" % epoch )
    return (best_validation_loss,bestmodelsda)
    
# -------------------------------------------------------------------------------------
def build_model(trainval_set, options):

    if options['retrain'] == 0:    

        if options['verbose'] > 4:
            print >> sys.stderr, ('... building the model')
        # construct the stacked denoising autoencoder class
    
        train_set_x, train_set_y = trainval_set

        #print train_set_x.get_value(borrow=True).shape
        #print train_set_y.shape.eval()

        n_train_batches  = train_set_x.get_value(borrow=True).shape[0]
        n_train_batches /= options['batchsize']

        #print >> sys.stderr, options['nclasses']
        #print >> sys.stderr, train_set_y.eval()
        #aakak
        
        sda = SdA(numpy_rng=options['numpy_rng'], theano_rng=options['theano_rng'],
                  n_ins = options['ndim'],
                  hidden_layers_sizes=options['hlayers'],
                  n_outs=options['nclasses'], n_outs_b=options['nclasses'], tau=None)

        if options['verbose'] > 4:
            print >> sys.stderr, ('... getting the pretraining functions')
        pretraining_fns = sda.pretraining_functions(train_set_x=train_set_x,
                                                    batch_size=options['batchsize'], tau=None)

    else:
        # Restoring to Finetuned values
        sda_reuse_pt_model = []
        for para_copy in options['sda_reuse_model'].params:
            sda_reuse_pt_model.append(para_copy.get_value())

        ###
        sda = options['sda_reuse_model']
        
        for ids in range(len(sda.params)):
            sda.params_b[ids].set_value(sda_reuse_pt_model[ids]) # set the value


        n_outs = sda.params_b[-2].get_value().shape[0]        
        if options['nclasses_source'] != options['nclasses']:
            print >> sys.stderr, ("Droping logistic layer...")
            sda.change_lastlayer(n_outs,options['nclasses'])

        # print sda.params[1].get_value()[-1]
        # print sda.params_b[1].get_value()[-1]
        # kkk

        ########### Reuse layer wise fine-tuning #################
        #print '... getting the finetuning functions'
        #print 'Reuse layer wise finetuning'
        pretraining_fns = None
        
    return (sda,pretraining_fns)


# -------------------------------------------------------------------------------------
def do_experiment( folds, options, nrun, sda_reuse_model ):

    modeloptions = {}
    param = list(itertools.product(
        options['nneurons'],
        options['hlayers'],
        options['pretraining_epochs'],
        options['training_epochs'],
        options['pretrain_lr'],
        options['finetune_lr'],
        options['batchsize'],
        options['threshold'],
        options['corruptlevels']
    )
    )

    print >> sys.stderr, param
    
    print >> sys.stderr, ('Number of combinations {0:03d}'.format(len(param)))

    step = 0
    besterror = numpy.inf
    # ---------------------------------------------------------------
    # cross validation
    # ---------------------------------------------------------------
    for k in range(0,len(param)):
        
        (nneurons,
         hlayers,
         pretraining_epochs,
         training_epochs,
         pretrain_lr,
         finetune_lr,
         batchsize,
         threshold,
         corruptlevels) = param[k]

        modeloptions = {
            'savetimes'          : False,
            'outputfolder'       : options['outputfolder'],
            'outputfolderres'    : options['outputfolderres'],
            'resolution'         : options['resolution'],
            'retrain'            : options['retrain'],
            'verbose'            : options['verbose'],
            'ndim'               : options['ndim'],
            'nclasses_source'    : options['nclasses_source'],
            'nclasses'           : options['nclasses'],
            'numpy_rng'          : options['numpy_rng'],
            'theano_rng'         : options['theano_rng'],
            'measure'            : options['measure'],
            'oneclass'           : options['oneclass'],
            'batchsize'          : batchsize,
            'hlayers'            : nneurons * numpy.ones((hlayers,)),
            # numpy.array(nneurons * numpy.ones((hlayers,)) * (1/(2*numpy.arange(1,hlayers+1)*1.)),dtype=numpy.int),
            'corruptlevels'      : corruptlevels*numpy.ones((hlayers,),dtype=numpy.float32),
            'pretraining_epochs' : pretraining_epochs,
            'training_epochs'    : training_epochs,
            'pretrain_lr'        : pretrain_lr,
            'finetune_lr'        : finetune_lr,
            'threshold'          : threshold,
            'sda_reuse_model'    : sda_reuse_model,
            'retrain_ft_layers'  : options['retrain_ft_layers'],
            'weight'             : options['weight'],
        }
        if k == 0:
            bestmodeloptions = copy.copy( modeloptions )

        if modeloptions['verbose'] > 2:
            print >> sys.stderr, "######################################################"
            print >> sys.stderr, "                     CROSS-VAL                        "
            print >> sys.stderr, "######################################################"
            print >> sys.stderr, modeloptions

        merror  = 0
        merrori = 0
        for cv in range(0,options['folds']):
            counter = step/(len(param)*options['folds']*1.)
            print >> sys.stderr, ('###### {t:0{format}.1f}% ({e:0.2f})'.format(format=5,t=counter*100,e=besterror) )
            trainset = folds[0]
            valset   = folds[1]
            testset  = folds[2]

            # print >> sys.stderr, sda_reuse_model
            (sda,pretraining_fns) = build_model(trainset[cv],modeloptions)
            # print >> sys.stderr, sda
            sda  = pretrain_finetune_model(sda,pretraining_fns,
                                           trainset[cv],
                                           valset[cv],
                                           modeloptions)[1]
            merrori = evaluate_model(sda,testset[cv],modeloptions)[0]
            merror  = merror + merrori
            # print >> sys.stderr, sda, sda_reuse_model
            
            step = step + 1
        merror = merror / options['folds']

        if merror < besterror:
            besterror        = merror
            bestmodeloptions = copy.copy( modeloptions )

    # print >> sys.stderr, "------------------------------"
    # -------------------------------------------------------------------
    # end of cross validation
    
    if modeloptions['verbose'] > 0:
        print >> sys.stderr, "######################################################"
        print >> sys.stderr, "                     TRAIN/TEST                       "
        print >> sys.stderr, "######################################################"
    print >> sys.stderr, (bestmodeloptions)
    trainset = folds[3]
    valset   = folds[4]
    testset  = folds[5]
    
    bestmodeloptions['savetimes'] = True 
    bestmodeloptions['nrun']      = nrun

    # print >> sys.stderr, sda_reuse_model
    start_time = time.clock()
    (sda,pretraining_fns) = build_model(trainset, bestmodeloptions)
    end_time = time.clock()
    
    pretrain_time = end_time - start_time

    start_time = time.clock()
    sda = pretrain_finetune_model(sda, pretraining_fns,
                                  trainset,
                                  valset,
                                  bestmodeloptions)[1]
    end_time = time.clock()
    finetune_time = end_time - start_time
    
    result = evaluate_model( sda, testset, bestmodeloptions )
    # print >> sys.stderr, sda, sda_reuse_model
    print >> sys.stderr, "time pretrain: {0:f} | time fine-tune: {1:f}".format(pretrain_time, finetune_time)
    result = result + ( pretrain_time, finetune_time )

    filename = '{0:s}/{1:05d}_{2:03d}_model.pkl.gz'.format(options['outputfolder'],nrun,string.atoi(options['resolution']))
    save_gzdata(filename, sda)

    filename = '{0:s}/{1:05d}_{2:03d}_options.pkl.gz'.format(options['outputfolder'],nrun,string.atoi(options['resolution']))
    save_gzdata(filename, bestmodeloptions)
    
    return result

# -----------------------------------------------------------------------------------------------------
def TL(
        source, target = None,
        path = '../gen_patches/dataset_noisy/', retrain = False, retrain_ft_layers = [1,1,1,1,1,1],
        outputfolder='backup',
        outputfolderres='backup_res',
        batchsize = 1000,
        sourcemodelspath = './'
):
    
    """
    TO DO: FINISH DOCUMENTATION
    """

    options = {
        'sourcemodelspath'  : sourcemodelspath,
        'outputfolder'      : outputfolder,
        'outputfolderres'   : outputfolderres,
        'verbose'           : 0,
        'viewdata'          : False,
        'trainsize'         : 0.6,
        'patchsize'         : 20,
        'measure'           : 'acc',
        'weight'            : 200,
        'datanormalize'     : True,
        # ---------- one-class learning
        'replicate'         : False,
        'oneclass'          : False,
        # ---------- source problem params
        'database_source'   : 'db2',
        'resolution_source' : source,
        'nclasses_source'   : 2, # TODO: do this automatically
        # ---------- target problem params
        'database_target'   : 'db2',
        'resolution_target' : target,
        # ---------- TL hyperparams
        'retrain'           : retrain,
        'retrain_ft_layers' : retrain_ft_layers,
        # ---------- hyperparams
        'nruns'             : 20,
        'folds'             : 3,
        'hlayers'           : [len(retrain_ft_layers) / 2],    # X hidden + 1 log layer
        'nneurons'          : [ 1000],     # range(500, 1001, 250),
        'pretraining_epochs': [ 1000],     # [200]
        'training_epochs'   : [ 3000],    # [100]
        'pretrain_lr'       : [ 0.01, 0.001],   #[ 0.01, 0.001],
        'finetune_lr'       : [ 0.1 , 0.01],  #[ 0.1, 0.01],
        'threshold'         : [0.8], #[ 0.5 , 0.6, 0.8], #numpy.arange(.5,1.01,.1),
        'batchsize'         : [ batchsize], #[100] or [1000] depending on the size of the dataset. 
        # ---------- end of hyperparams
        'corruptlevels'     : [0.1], #numpy.arange(0.1, 0.4, 0.1)
    }
    
    print >> sys.stderr, (options), "\n"

    # -------------------------------------------------------------------------------
    datasetpath = path
    # print argv
    # print datasetpath
    # print retrain_ft_layers
    # alaallala

    # -------------------------------------------------------------------------------
    # load dataset
    if options['retrain'] == 1:
        options['database']   = options['database_target']
        options['resolution'] = options['resolution_target']
    else:
        options['database']   = options['database_source']
        options['resolution'] = options['resolution_source']

    (dataset, ndim, nclasses)   = load_data( datasetpath, options )
    options['ndim']     = ndim
    options['nclasses'] = nclasses

    # --------------------------------------------------------------------------------------------
    for nrun in range(1,options['nruns']+1):
        print >> sys.stderr, ("### {0:03d} of {1:03d}".format(nrun,options['nruns']))
        options['numpy_rng']  = numpy.random.RandomState(nrun)
        options['theano_rng'] = RandomStreams(seed=nrun)

        # --------------
        # generate folds
        folds = gen_folds( dataset, options, nrun )    
        # continue
        
        if options['retrain'] == 1:
            filename = "{0:s}/{1:05d}_{2:03d}_model.pkl.gz".format(options['sourcemodelspath'], nrun,
                                                              string.atoi(options['resolution_source']))
            print >> sys.stderr, ":: Loading model {0:s}...\n".format(filename)
            sda_reuse_model = load_savedgzdata ( filename )

            #print sda_reuse_model.logLayer.W.get_value()
            #print sda_reuse_model.logLayer.W.get_value()
            #kkk
            
        else:
            sda_reuse_model = None

        # ----------------------------------------------------------------------------
        results = do_experiment( folds, options, nrun, sda_reuse_model )
        # ----------------------------------------------------------------------------
    
        # --------------------------------------------------
        filename = '{0:s}/res_{1:05d}_{2:03d}.pkl.gz'.format(options['outputfolderres'],nrun,string.atoi(options['resolution']))
        save_results(filename,results)
        
    #-------------end testing the SdA


if __name__ == '__main__':
    if len(os.sys.argv) < 2:
        print_usage()
        os.sys.exit(-1)

    print os.sys.argv

    # execute experiment
    TL(os.sys.argv)
