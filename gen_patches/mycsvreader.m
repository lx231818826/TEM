function data = mycsvreader(filename)

f = fopen(filename);

header = textscan(f, '%s', 4, 'delimiter', ',');
lines = textscan(f, '%d %d %d %d', ...
                 'delimiter', ',', 'BufSize', 2^16);

% data is a struct containing all the instances of the dataset
data = cell2struct(lines, header{1}, 2); 

fclose(f);

return