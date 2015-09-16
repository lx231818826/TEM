#!/usr/bin/python

import os, string, sys

lib_path = os.path.abspath('./TL/')
sys.path.append(lib_path)

from data_handling import load_savedgzdata

basepath_15000 = sys.argv[1]
resolution_15000 = '15000'
basepath_50000 = sys.argv[2]


for nrun in range(1,21):
    pathids0 = '{0:s}/{1:05d}_{2:05d}_test_ids.pkl.gz'.format(basepath_15000,nrun,string.atoi(resolution_15000))
    pathids1 = '{0:s}/{1:05d}_{2:05d}_test_ids.pkl.gz'.format(basepath_50000,nrun,string.atoi(resolution_15000))

    print >> sys.stderr, 'Loading ' + pathids0 + '...'
    ids0 = load_savedgzdata(pathids0)
    print >> sys.stderr, 'Loading ' + pathids1 + '...'
    ids1 = load_savedgzdata(pathids1)
    print >> sys.stderr, ids0
    print >> sys.stderr, ids1

    #raw_input()

