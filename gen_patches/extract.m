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

% --------------------------------------------------------------------
% extract                         
% will read images from a base path dir and based on the
% corresponding annotations will extract a patch with a predefined
% size

function db = extract(bdpath,annotators,options)
debug = options.debug;

descriptor_positive = uint8([]);
descriptor_negative = uint8([]);

% windows size (square)
wsize = 20;

% get all tif or jpg files to work with
fileStack = listAllFiles(bdpath);

% loads images and respective annotations
for file = 1:length(fileStack)
    % loads image file
    imgname = fullfile(fileStack(file).path,fileStack(file).filename);
    idxnamestrip  = strfind(fileStack(file).filename,'.');

    msg = sprintf('Loading %s\n',imgname);
    printMsg(msg,debug);
    
    img     = imread(imgname);

    if options.toresize
        img = imresize(img,options.resize);
    end
    
    [nrows,ncols,nch] = size(img);
    if nch > 1
        img = rgb2gray(img);
    end
    
    % init
    dpos = uint8([]);
    dneg = uint8([]);

    % loads annotation file
    for ann=1:length(annotators)
        annfilename = [fileStack(file).filename(1:idxnamestrip-1) ,'.csv'];
        annfilename = fullfile(fileStack(file).path,'annotation',annotators{ann},annfilename);

        msg = sprintf('Annotations file %s\n',annfilename);
        printMsg(msg,debug);
            
        annotation = mycsvreader(annfilename);
        
        % for each point
        nelem = length(annotation.xtopleft);
        
        % get central positions
        GTx = double(annotation.xtopleft+annotation.xbottomright)/2; 
        % xx coordinate
        GTy = double(annotation.ytopleft+annotation.ybottomright)/2; 
        % yy coordinate
            
        if options.toresize
            GTx = GTx*options.resize;
            GTy = GTy*options.resize;
        end
        
        GTx = round(GTx);
        GTy = round(GTy);
        
        msg = sprintf('There are %d points..\n',nelem);
        printMsg(msg,debug);
        
        for pt = 1:nelem
            % positive patches
            
            % ------------------------------------------------------------------------
            % get window indexes
            [mpatch,isoutlimits] = getPatch(img,wsize,GTx(pt),GTy(pt),ncols,nrows,GTx,GTy,options,true);
        
            if isoutlimits == 1
                continue
            end                
            
            dpos = [dpos; file mpatch];
            
            % negative patches
            % ------------------------------------------------------------------------
            while 1
                ri = randi(options.randstream,nrows,1,1);
                ci = randi(options.randstream,ncols,1,1);

                % there is some point that it is close to our random
                % location that it should 
                if any( pdist2([ci,ri],[GTx,GTy]) < wsize*4 )
                    continue
                end
                [mpatch,isoutlimits] = getPatch(img,wsize,ci,ri,ncols,nrows,GTx,GTy,options);
                
                if isoutlimits == 1
                    continue
                end
                
                dneg = [dneg; file mpatch];
                break
            end
            % end while
        end
        % end markers
    end
    % end annotators
    
    descriptor_positive = [descriptor_positive; dpos];
    descriptor_negative = [descriptor_negative; dneg];
    
end 
% end of list of files
db = struct();
db.positive = descriptor_positive;
db.negative = descriptor_negative;

return


%%                                                          
function [mpatch,isoutlimits] = getPatch(img,wsize,x,y,ncols,nrows,GTx,GTy,options,GTgiven)
mpatch = [];

if nargin == 9
    GTgiven = false;
end

hwsize  = wsize/2;
%hwsize2 = wsize/4;

if GTgiven
    x = round(x + (-hwsize + hwsize*2*rand));
    y = round(y + (-hwsize + hwsize*2*rand));
end

cols = (x-hwsize):(x+hwsize-1);
rows = (y-hwsize):(y+hwsize-1);
[cols,rows] = meshgrid(cols,rows); 
        
isoutlimits  = ( min(cols(:)) < 1 | max(cols(:)) > ncols | ...
                 min(rows(:)) < 1 | max(rows(:)) > nrows );
if isoutlimits == 1
    return
end

if options.noisy == 0
    if sum( pdist2([x,y],[GTx,GTy]) < wsize*2) > 1
        isoutlimits = 1;
        return
    end
end


% get patch
ind = sub2ind([nrows, ncols],rows,cols);
mpatch = img(ind);

mpatch = mpatch(:)';

return 
