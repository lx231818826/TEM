function data = mycsvreader(filename,options)

f = fopen(filename);

header = textscan(f, '%s', 4, 'delimiter', ',');
lines = textscan(f, '%d %d %d %d', ...
                 'delimiter', ',', 'BufSize', 2^16);

% data is a struct containing all the instances of the dataset
data = cell2struct(lines, header{1}, 2); 
data.xtopleft = data.xtopleft / options.resize;
data.ytopleft = data.ytopleft / options.resize;
data.xbottomright = data.xbottomright / options.resize; 
data.ybottomright = data.ybottomright / options.resize;

fclose(f);

return