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
import glob, numpy, sys
import cPickle as pickle
import gzip

def usage():
   print "./process.py resultsdir" 

def process_files(files):
   allperf = []
   for file in files:
      # print file
      
      f = gzip.open( file, "rb" )
      perf = pickle.load( f )
      f.close()
      allperf.append(perf)

   return allperf

def times(argv):
   basepath  = argv[1]
   files = glob.glob( basepath + "times_pr*")
   files.sort()
   
   allperf = process_files(files)
   # print allperf
   if len(allperf) > 0:
      print "basepath pretraining times: {0:f} +/- {1:f}".format(numpy.mean(allperf),numpy.std(allperf))

   alltimes = numpy.array( allperf )
   if len( alltimes ) == 0:
      alltimes = 0
   
   files = glob.glob( basepath + "times_fn*")
   files.sort()

   allperf = process_files(files)
   # print allperf
   
   print "basepath finetuning times: {0:f} +/- {1:f}".format(numpy.mean(allperf),numpy.std(allperf))

   alltimes = alltimes + numpy.array(allperf)

   print "all times: {0:f} +/- {1:f}".format(numpy.mean(alltimes),numpy.std(alltimes))
   
def perf(argv):
   basepath  = argv[1]
   files     = glob.glob( basepath + "res_*")
   files.sort()

   allperf = process_files(files)
   allperf = map(lambda x: 100-x[0]*100,  allperf )

   print "basepath perf. results: {0:f} +/- {1:f}".format(numpy.mean(allperf),numpy.std(allperf))
   

 
if __name__ == "__main__":
   if len(sys.argv) != 2:
      usage()
      sys.exit()
      
   perf(sys.argv)
   times(sys.argv)

