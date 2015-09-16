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

% ----------------------------------------------------------------------------
% genPatches
% this function will generate our (patch) database
% for training a sda or other ML algorithm

function genPatches(basepath,annotators,database)

STREAM = RandStream('mrg32k3a');
RandStream.setGlobalStream(STREAM);


% sanity check
if nargin == 0
    % small trick. i am lazy
    basepath = '../../';
end

% --------------------------------------------------------
%imgs  = {'imgs_nanoparticles/15000','imgs_nanoparticles/20000','imgs_nanoparticles/30000','imgs_nanoparticles/50000'}; %,'imgs/20000','imgs/30000','imgs/50000'};
%imgs  = {'imgs_nanoparticles/15000','imgs_nanoparticles/20000','imgs_nanoparticles/30000','imgs_nanoparticles/50000'}; %,'imgs/20000','imgs/30000','imgs/50000'};
imgs  = {'imgs_nanoparticles/30000'};

options = struct();
options.debug    = 0;
options.toresize = 1;
options.resize   = .5;
options.noisy    = 1;
options.randstream = STREAM;

fprintf(1,'Generating patches..');
for imgidx=1:length(imgs) 
    imgdir = imgs{imgidx};
    bdpath = fullfile( basepath,imgdir,database );

    mpatches = extract(bdpath,annotators,options);

    filename = sprintf('db_%s_noisy_%06.0f.mat',database,str2num(imgdir(end-4:end)))

    msg = sprintf('Saving databse in %s folder.\n',filename);
    printMsg(msg,options.debug);
    
    save(filename,'mpatches','-v7.3')
    
end
fprintf(1,'..Done\n');

return
