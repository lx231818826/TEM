function icyDetections = getIcyDetections(fileName,stepScales)

f = fopen(fileName);

if f == -1
    error('File inexistent.')
end

for t=1:27+stepScales, fgetl(f); end
header = textscan(f, '%s', 9, 'delimiter', ';');
lines = textscan(f, '%f %f %f %f %f %f %f %f %f', ...
                 'delimiter', ';', 'BufSize', 2^16);
fclose ( f );
header{1}{1} = 'Detection';
header{1}{7} = 'minintensity';
header{1}{8} = 'maxintensity';
header{1}{9} = 'avgintensity';

data = cell2struct(lines,header{1},2);

% Load excel file
% data = xlsread(fullfile(fileNamePath,fileName(i).name));

% Obtain detection location

nPoints = length(data.Detection);
for k = 1:nPoints
    icyDetections(k).x = data.x(k);
    icyDetections(k).y = data.y(k);
end

return