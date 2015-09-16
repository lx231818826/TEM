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
import sys
import cPickle as pickle
import gzip

def load_savedgzdata(filename):
    f = gzip.open(filename, 'rb')
    data = pickle.load(f)
    f.close()
    return data

def save_gzdata(filename,data):
    f = gzip.open(filename, 'wb')
    data = pickle.dump(data,f)
    f.close()
    return data

def print_file(filename,string):
    f = open(filename, 'a')
    f.write(string)
    f.close()

def load_saveddata(filename):
    data = pickle.load(open( filename, "rb" ) )
    return data
        
def save_data(filename, model):
    print >> sys.stderr, ("Saving: " + filename)
    pickle.dump( model, open( filename, "wb" ) )
    
def save_results(filename,res):
    merror = res[0]
    print >> sys.stderr, ("Test error: {t:0{format}.1f}%\n".format(format=5,t=merror*100.))
    #pickle.dump( res, open( filename, "wb" ) )
    save_gzdata(filename,res)
