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
% ------------------------------------------------------------------------------------

data.partiDetMethod = 'log_detector';
data.train_ids      = [1];
data.val_ids        = [1; 4; 7];
data.trainfinal_ids = [1];
data.valfinal_ids   = [1];
data.test_ids       = [5 10  8 21  4  1 22 16 13];
data.resolution     = '15000_1';

script_sda_detector( data )