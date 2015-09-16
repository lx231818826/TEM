
%%                                                                                                         
% ------------------------------------------------------------------------
function obtainIcyResults(basepath,folderDirectory,distMin,options,scaleList,szThList,stepScales)

% number of points for precision/recall graph
precRecPoints = 50:10:110;

% Main folder under analysis
icypath = fullfile(basepath,folderDirectory,'Icy');
imgList = dir(icypath);

fileStack = {};
nImgs     = 1;
% For each image
for file = 1:length(imgList)
    if ( strcmp(imgList(file).name,'.') || ...
         strcmp(imgList(file).name,'..') )
        continue
    end

    fileStack{nImgs} = imgList(file).name;
    nImgs = nImgs + 1;
end
nImgs = length(fileStack);


% Get manual annotation
nAvgGoldPartiAnn = zeros(1,nImgs);
dbpath           = fullfile(basepath,folderDirectory);
for img=1:nImgs
    annfilename = [fileStack{img},'.csv']
    annotation  = getAnnotations(options,dbpath,annfilename);
    
    nAnnotators = length(options.annotators);
    nAvgGoldPartiAnn_tmp = 0;
    for ann=1:nAnnotators
        nAvgGoldPartiAnn_tmp = nAvgGoldPartiAnn_tmp + length(annotation{ann}.xtopleft);
    end
    nAvgGoldPartiAnn(img) = nAvgGoldPartiAnn_tmp / nAnnotators;
    
end

ROC_precision = zeros(length(scaleList), length(szThList), length(precRecPoints));
ROC_recall    = zeros(length(scaleList), length(szThList), length(precRecPoints));

                    
% % Print Roc curves
% close all
% colors = 'grycmkb'; colorid = 0; curvesColor = [];
% fig = figure;
% ylabel('Recall (%)'); xlabel('Precision (%)'); hold on
                    
% For each scale
for scaleIDX = 1:length(scaleList)  
        
        % For each size filter
        for szThIDX = 1:length(szThList)

            % Get points to plot the precision recall curve...
            precision = [];
            recall = [];
            for precRecPointsIDX = 1:length(precRecPoints)
                
                % Measure error
                TP_T = zeros(1,nImgs);
                FP_T = zeros(1,nImgs);
                FN_T = zeros(1,nImgs);
                Precision_T = zeros(1,nImgs);
                Recall_T    = zeros(1,nImgs);
                
                % ... For each image
                for file = 1:nImgs
                    
                    % Get annotation from each user to evaluate results
                    annfilename = [fileStack{file},'.csv'];
                    annotation  = getAnnotations(options,dbpath,annfilename);
                    
                    scale = sprintf('Scale %01d',scaleList(scaleIDX));
                    sizen = sprintf('size %01d',szThList(szThIDX));
                    precRecPointsElem = sprintf('save %03d',precRecPoints(precRecPointsIDX));
                    
                    icyFullpath = ... 
                        fullfile(icypath,fileStack{file},scale,sizen,precRecPointsElem);

                    filename = sprintf('%s.csv',fileStack{file});
                    saveName = fullfile(icyFullpath,filename);
                    fprintf(1,'Loading %s\n',saveName);

                    % Get icy detection results
                    icyDetections = getIcyDetections(saveName,stepScales(scaleIDX));
                    
                    % Compare automatic detections with manual detections
                    [TP, FP, FN] = performEvaluation(icyDetections, annotation, distMin);
                    TP_T(file) = TP;
                    FP_T(file) = FP;
                    FN_T(file) = FN;
                    
                    %manualDetections = length(annotation.xtopleft);
                    %icyDetections = length(icyDetections);
                    Precision_T(file) = (TP/(TP+FP+0.000000001)); % True positives / number of automatic detections
                    Recall_T(file)    = (TP/nAvgGoldPartiAnn(file)); % True positives / number of existing gold nanoparticles
                    
                end
                
                % Save data
                pathtosave = fullfile(basepath,folderDirectory,'resultado','nanoParticlesDetectionResult_icy');
                dataname   = sprintf('%s_scale_%01d_size_%01d_th_%03d',pathtosave,...
                                     scaleList(scaleIDX),szThList(szThIDX),precRecPoints(precRecPointsIDX));
                fprintf('\nSaving results data in %s\n',dataname);
                save(dataname,'TP_T', 'FP_T', 'FN_T', 'Precision_T', 'Recall_T', 'icyDetections','-v7.3')

                % Final Results
                ROC_precision(scaleIDX,szThIDX,precRecPointsIDX) = mean(Precision_T);
                ROC_varPrec(scaleIDX,szThIDX,precRecPointsIDX)   = std(Precision_T);
                ROC_recall(scaleIDX,szThIDX,precRecPointsIDX)    = mean(Recall_T);
                ROC_varReca(scaleIDX,szThIDX,precRecPointsIDX)   = std(Recall_T);
                fprintf(1,'Recall - %3.2f | Precision - %3.2f\n',mean(Recall_T),mean(Precision_T));
                
%                 Precision_T
%                 Recall_T
%                 precision(end+1) = mean(Precision_T);
%                 recall(end+1)    = mean(Recall_T);
            end
            
%             % Plot curve
%             colorid = colorid + 1;
%             hold on
%             plot(precision(:),recall(:),colors(mod(colorid,7)+1),'LineWidth',3);
%             pause(0.5)
%             curvesColor{end+1} = ['Scale: ',num2str(scaleList(scaleIDX)), ', Size: ',num2str(szThList(szThIDX))];
            
        end

end
% % Figure legend
% legend(curvesColor,'Location','BestOutside')


% Save global data
dataname = fullfile(basepath,folderDirectory,'resultado','GlobalNanoParticlesDetectionResult_icy');
save(dataname,'ROC_precision', 'ROC_varPrec', 'ROC_recall', 'ROC_varReca','-v7.3')

return