
%% reconstruct                         
% script to check if the dataset is well constructed

% GNU LGPL license

% :: authors ::
% Ricardo Sousa
% rsousa _at_ rsousa.org

function reconstruct()

mpatches = load('dataset_noisy/db_db2_noisy_050000.mat');
mpatches = mpatches.mpatches;

mpatches

fprintf(1,'Positives...\n');
getReconstruction(mpatches.positive,1)
fprintf(1,'Negatives...\n');
getReconstruction(mpatches.negative,0)

return

function getReconstruction(mpatches,flag)

nelem = size(mpatches,1);
%idx   = randperm(nelem);
idx = 1:nelem;

for n=1:nelem
    mpatch = mpatches(idx(n),2:end);
    wsize  = sqrt(length(mpatch));
    mpatch = reshape(mpatch,wsize,wsize);
    
    % figure, imshow(mpatch,[])
    % pause
    % close all

    filename = sprintf('samples/%d_%03d.jpg',flag,n);
    imwrite(mpatch,filename);
    
    if n > 1000
        break
    end
    %if randi(2,1)-1 == 1
    %    break
    %end
end

return