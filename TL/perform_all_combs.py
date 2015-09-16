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

import numpy, os, string, sys
from main import TL

def create_folder(method,foldername):
    outputfolder    = 'final_results_{0:s}/models/{1:s}'.format(method,foldername)
    command         = 'mkdir -p ' + outputfolder
    print >> sys.stderr, command
    os.system(command)

    outputfolderres = 'final_results_{0:s}/results/{1:s}'.format(method,foldername)
    command         = 'mkdir -p ' + outputfolderres
    print >> sys.stderr, command
    os.system(command)

    return (outputfolder, outputfolderres)

def run_TL(i,layers,source,target,batchsize):
    
    command0        = ''.join(str( v ) for v in layers)

    sourcemodelspath  = '../final_results/baseline_resized/{0:s}/models/res_baseline_resized_{0:s}_111111/'.format(source)
    
    (outputfolder, outputfolderres) = create_folder('TL','res_tl_resized_{0:s}_{1:s}_{2:s}'.format(source,target,command0))
    TL(
        source, target, path = '../gen_patches/dataset_noisy/', retrain=True, retrain_ft_layers = layers,
        outputfolder = outputfolder, outputfolderres = outputfolderres,
        batchsize    = batchsize,
        sourcemodelspath = sourcemodelspath
    )


# -----------------------------------------------------------------------------------------------
def run_baseline(i,layers,batchsize,source):
    
    command0        = ''.join(str( v ) for v in layers)
    (outputfolder, outputfolderres) = create_folder('baseline','res_baseline_resized_{0:s}_{1:s}'.format(source,command0))
    
    TL(
        source, target = None, path = '../gen_patches/dataset_noisy/', retrain_ft_layers = layers,
        outputfolder = outputfolder, outputfolderres = outputfolderres,
        batchsize    = batchsize
    )

# -----------------------------------------------------------------------------------------------
def run_baseline_resized(i,layers,batchsize,source):
    
    command0        = ''.join(str( v ) for v in layers)
    (outputfolder, outputfolderres) = create_folder('baseline','res_baseline_resized_{0:s}_{1:s}'.format(source,command0))
    
    TL(
        source, target = None, path = '../gen_patches/dataset_noisy_resized/', retrain_ft_layers = layers,
        outputfolder = outputfolder, outputfolderres = outputfolderres,
        batchsize    = batchsize
    )
    

# -----------------------------------------------------------------------------------------------
def perform_all_baseline_combs():
    #sources = ['15000','20000','30000','50000']
    #possible_batchsizes = [1000, 100, 100, 100]

    sources = ['50000']
    possible_batchsizes = [100]

    #sources = ['20000']
    #possible_batchsizes = [100]

    possible_layers     = [3] #range(3,5)
    
    for source in range(0,len(sources)):
        batchsize = possible_batchsizes[source]
        
        for i in range(0,len(possible_layers)):
            layers = numpy.ones((possible_layers[i]*2),dtype=numpy.uint8)
            run_baseline(i,layers,batchsize,sources[source])

# -----------------------------------------------------------------------------------------------
def perform_all_baseline_resized_combs():
    sources = ['50000']

    possible_batchsizes = [100]
    possible_layers     = [3] # range(3,5)
    
    for source in range(0,len(sources)):
        batchsize = possible_batchsizes[source]
        
        for i in range(0,len(possible_layers)):
            layers = numpy.ones((possible_layers[i]*2),dtype=numpy.uint8)
            run_baseline_resized(i,layers,batchsize,sources[source])
            
# -----------------------------------------------------------------------------------------------
def convert(layers):
    layers = map(lambda x: [x,x], layers)
    layers = [item for sublist in layers for item in sublist]
    layers = list(layers)
    return layers

def perform_all_tl_combs():
    source = '30000'
    target = {'db': '15000', 'batchsize': 1000}

    retrainlayers = [0,0,0]
    for i in range(len(retrainlayers)-1,0,-1):
        retrainlayers[i] = 1
        layersRep = convert(retrainlayers)
        run_TL(i,layersRep,source,target['db'],target['batchsize'])
        
    retrainlayers = [0,0,0]
    for i in range(0,len(retrainlayers)):
        retrainlayers[i] = 1
        layersRep = convert(retrainlayers)
        run_TL(i,layersRep,source,target['db'],target['batchsize'])

if __name__=="__main__":
    perform_all_baseline_combs()
    #perform_all_baseline_resized_combs()
    #perform_all_tl_combs()
