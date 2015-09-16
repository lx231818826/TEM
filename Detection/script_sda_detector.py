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
import numpy, sys, string, os
lib_path = os.path.abspath('../TL/')
sys.path.append(lib_path)
from data_handling import load_savedgzdata
import cPickle as pickle


def script_sda_detector(resolution):

    import pymatlab
    session = pymatlab.session_factory()
    
    nruns      = 20
    partiDetMethod = 'log_detector'
 
    for nrun in range(18,nruns+1):
    
        basefilename = '../final_results/baseline_resized/{0:05d}/models/res_baseline_resized_{0:05d}_111111/{1:05d}_{2:03d}_'.format(resolution,nrun,resolution)

        trainfilename      = basefilename + 'train_ids.pkl.gz'
        valfilename        = basefilename + 'val_ids.pkl.gz'
        trainfinalfilename = basefilename + 'trainfinal_ids.pkl.gz'
        valfinalfilename   = basefilename + 'valfinal_ids.pkl.gz'
        testfilename       = basefilename + 'test_ids.pkl.gz'

        # [0,max_ids] -> [1,max_ids+1]
        train_ids      = load_savedgzdata(trainfilename)+1
        val_ids        = load_savedgzdata(valfilename)+1
        trainfinal_ids = load_savedgzdata(trainfinalfilename)+1
        valfinal_ids   = load_savedgzdata(valfinalfilename)+1
        test_ids       = load_savedgzdata(testfilename)+1

        print >> sys.stderr, train_ids
        print >> sys.stderr, val_ids
        print >> sys.stderr, trainfinal_ids
        print >> sys.stderr, valfinal_ids
        print >> sys.stderr, test_ids

        session.putvalue('partiDetMethod',partiDetMethod)
        session.putvalue('resolution',str(resolution) + '_' + str(nrun))
        session.putvalue('train_ids',train_ids)
        session.putvalue('val_ids',val_ids)
        session.putvalue('trainfinal_ids',trainfinal_ids)
        session.putvalue('valfinal_ids',valfinal_ids)
        session.putvalue('test_ids',test_ids)
        
        mscript = """
        data = struct();
        data.partiDetMethod = partiDetMethod;
        data.resolution     = resolution;
        data.train_ids      = train_ids;
        data.val_ids        = val_ids;
        data.trainfinal_ids = trainfinal_ids;
        data.valfinal_ids   = valfinal_ids;
        data.test_ids       = test_ids;
        res = script_sda_detector( data )
        """
        
        session.putvalue('MSCRIPT', mscript)
        session.run('eval(MSCRIPT)')
        res = session.getvalue('res')
        print res

if __name__=="__main__":
    script_sda_detector(15000)
            


