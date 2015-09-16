% Ricardo Sousa
% rsousa at rsousa.org

% Copyright 2014 Ricardo Sousa

% This file is part of NanoParticles.

% NanoParticles is free software: you can redistribute it and/or modify 
% it under the terms of the GNU General Public License as published
% by the Free Software Foundation, either version 3 of the License,
% or (at your option) any later version.

% NanoParticles is distributed in the hope that it will be useful, but 
% WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU 
% General Public License for more details.

% You should have received a copy of the GNU General Public License 
% along with NanoParticles. If not, see http://www.gnu.org/licenses/.

%%                                                                         
% This function executes all experiments
function execute_all( partiDetMethod, resolution, train_ids, val_ids, ...
                      trainfinal_ids, valfinal_ids, test_ids, nrun )
% ----------------------------------------------------
% DO NOT CHANGE AFTER THIS LINE
% EVERYTHING YOU DO AFTER THIS LINE IS YOUR RESPONSABILITY
% ----------------------------------------------------

% base dir
basepath   = '../';
options = struct();
options.debug     = true;
options.nfolds    = 3;
options.resize    = 1; % resize images by half
options.resultdir = 'resultado_sae';
options.withMask  = false;

if nargin == 1
    options.userParamGiven = false;
    imgs       = {'../images/15000/'};
    dists      = [4 6 10 10]; % distances for each size
    annotators = {'user'};
    database   = 'db2';
    
    % icy stuff
    stepScalesA= {[0, 0],[1,1],[0,1],[0, 1]};
    scale      = {[3 7], [7 13], [7 13], [7 13]};
    sizes      = {[0 10 15], [0 10 15], [0 10 15], [0 10 15 100]};

    options.resolution    = 15000;
    options.nrun          = 1;
    options.train_ids     = [1];
    options.val_ids       = [1, 1, 1; 2, 2, 2; 3, 3, 3];
    options.trainfinal_ids= [1];
    options.valfinal_ids  = [1, 1, 1; 2, 2, 2; 3, 3, 3];
    options.test_ids      = [4];

else
    database   = 'db2';
    imgs       = {fullfile('../images/', num2str(resolution))};
    annotators = {'user'};
    options.resolution     = resolution;
    options.userParamGiven = true;
    options.train_ids      = train_ids;
    options.val_ids        = val_ids;
    options.trainfinal_ids = trainfinal_ids;
    options.valfinal_ids   = valfinal_ids;
    options.test_ids       = test_ids; 
    options.nrun           = nrun;
end

options.partiDetMethod = lower(partiDetMethod);
options.annotators     = annotators;

options
for i=1:length(options.test_ids)
    fprintf(2,':: %d\n',options.test_ids(i));
end



for imgidx=1:length(imgs) 
    if isempty( imgs{imgidx} )
        continue
    end
    imgdir  = fullfile( imgs{imgidx}, database  );
    distMin = 0; % NOT IMPLEMENTED ON ICY
        
    switch( options.partiDetMethod )
        
      case {'log_detector','hough'}
        % Gold nanoparticle automatic detection
        run(basepath,imgdir,options);
      
      case 'icy'
        error('_deprecated_');
        scaleIdx  = scale{imgidx};
        sizesIdx  = sizes{imgidx};
        
        obtainIcyResults(basepath,imgdir,distMin,options,scaleIdx,sizesIdx,stepScalesA{imgidx});

      otherwise
        errmsg = sprintf('************* Method unknown: %s.', options.partiDetMethod);
        error( errmsg );
    end
end

return
