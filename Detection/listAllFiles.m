function fileStack = listAllFiles(directory)
% searches for duplicated files, starting from a root directory

fileStack = [];
n    = 0;
list = dir(directory); 

for f = 1:length(list)
   if (strcmp(list(f).name,'.') > 0) || (strcmp(list(f).name,'..') >0)
       % do nothing
%    elseif (list(f).isdir>0)
%        % recurrently calls the listing of this directory
%        fileStack2 = listAllFiles([directory '\' list(f).name]);
%        fileStack  = cat(2,fileStack,fileStack2);
   elseif (~isempty(strfind(list(f).name,'.tif')) || ~isempty(strfind(list(f).name,'.jpg')))
       
       % reads in the file info into the fileStack
       if 0 == n && isempty(fileStack)
           fileStack = struct('path','','filename','','size',0);
       end
       
       n = length(fileStack) + (n~=0); % trick not to add 1 when n == 0 
       
       fileStack(n).filename = list(f).name; %#ok<*AGROW>
       fileStack(n).size = list(f).bytes;
       fileStack(n).path = directory;
   end
end