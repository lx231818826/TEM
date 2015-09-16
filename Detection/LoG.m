% Tiago Esteves, Pedro Quelhas
% dee11017 @ fe.up.pt

% Copyright 2014 Tiago Estes, Pedro Quelhas

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
function [sp, block] = LoG(im, Rmin, Rmax,step,sigma,th,disk_size,mask,signal)
%
% signal: -1 white color cells
%         +1 black color cells
%          2 abs
%

if isempty(mask)
    mask = ones(size(im));
end

img = double(im);

[rows, cols] = size(im);

block = [];
G     = fspecial('gaussian', round(5*sigma), sigma);
img   = imfilter(img, G,'symmetric');

for r = Rmin:step:Rmax
    fprintf(1,'.');
    s=round(r/1.5); %ajustar o filtro para este raio
    h = (s^2) * fspecial('log',round(s*5),s); % minus just so max is what we want
    if signal == 2
        h = abs(h);
    else
        h = signal*h;
    end
    
    result = imfilter(img,h,'replicate');
    
    % creates a scale space block to find location and scale of nuclei
    block = cat(3,block, result);
end

[Cxy, indx] = max(block,[],3);

indx = (indx-1).*step + 1;

% 
G     = fspecial('gaussian', 10, 1.4);
Cxy_g = imfilter(Cxy, G,'symmetric');

ir_max = imregionalmax(Cxy_g);
ir_max = ir_max.*Cxy_g;
ir_max = ir_max.*mask;
ir_med = imdilate(ir_max,strel('disk',disk_size,0));
ir_max = (ir_med == ir_max).*ir_max;
[r, c, resp] = find(ir_max>th);

% if frame_number > 130
%     figure, imshow(Cxy,[])
%     figure, imshow(Cxy_g,[])
%     figure, imshow(ir_max,[])
% end

sp = struct('radius',[],'x',[],'y',[],'resp',[]);
n = 0;
for p = 1:length(r)
    if mod(p,50) == 0
        fprintf(1,'.');
    end
    rdx = indx(r(p),c(p)) + Rmin - 1;
    if (c(p) - rdx) > 0 && (r(p) - rdx) > 0  && (c(p) + rdx) < cols && (r(p) + rdx) < rows
        % if c(p) > 0 && r(p) > 0  && c(p) < cols && r(p) < rows
        n = n + 1;
        sp(n).x      = c(p);
        sp(n).y      = r(p);
        sp(n).radius = rdx;
        sp(n).resp   = ir_max(r(p),c(p));
        
   end
end

% ORDERING OF SP BY RESPONSE OF FILTER
resp = cat(1,sp(:).resp);
[val, order] = sort(-resp); %#ok<*ASGLU>
sp = sp(order);

% max(ir_max(:))

return



