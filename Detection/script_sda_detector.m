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

function res = script_sda_detector( data )

idx = strfind(data.resolution,'_');
fprintf(2,'resolution: %s\n',data.resolution);

resolution = str2num(data.resolution(1:idx-1));
nrun       = str2num(data.resolution(idx+1:end));

data.resolution = resolution;
data.nrun       = nrun;

fprintf(2,'-->%d\n',data.resolution);
fprintf(2,'-->%d\n',data.nrun);

fprintf(2,'entering..\n')
execute_all(data.partiDetMethod,data.resolution, data.train_ids, ...
            data.val_ids, data.trainfinal_ids, data.valfinal_ids, data.test_ids, data.nrun)

res = size(data.val_ids);

return

