
function debug_annotations()

database = '';
basepath = '..';
imgs = {'images/15000'};
experts = {'user'};

for imgidx=1:length(imgs)
    dirname  = fullfile(basepath,imgs{imgidx},database);
    
    fileStack = listAllFiles(dirname);
   
    
    for fileidx =1:length(fileStack)
        % reads the image
        img = imread( fullfile(fileStack.path,...
                               fileStack.filename) );

        % reads the annotation for all experts
        extidx = strfind(fileStack.filename,'.')
        annot_filename = strcat( fileStack.filename(1:extidx),...
                                   'csv')
        
        annotations = cell(1,length(experts));
        for expidx = 1:length(experts)
            expert = experts{expidx};
            annot_filename_fullpath = ...
                fullfile(fileStack.path,...
                         'annotation',...
                         expert,...
                         annot_filename);
            
            annotations{expidx} = mycsvreader( annot_filename_fullpath );

        end
        
        %plots the data (x,y)
        plotData(img,annotations);
        return
    end
    
    
end
return


%---------------------------------------------------------------
function plotData(img,annotations)
colors = {[1 0 0], [0 1 0], [0 0 1]};

figure, imshow(img)
hold on

for annidx=1:length(annotations)
    ann_data = annotations{annidx};
    
    ann_data.xbottomright = double(ann_data.xbottomright);
    ann_data.ybottomright = double(ann_data.ybottomright);
    ann_data.xtopleft     = double(ann_data.xtopleft);
    ann_data.ytopleft     = double(ann_data.ytopleft);
    
    x = ( ann_data.xbottomright - ann_data.xtopleft )/2 + ann_data.xtopleft;
    y = ( ann_data.ybottomright - ann_data.ytopleft )/2 + ann_data.ytopleft;

    plot(x,y,'+','Color',colors{annidx},'MarkerSize',5)
    
end

return