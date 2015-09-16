function run(basepath,imgdir,options)
%close all

% Choose files to analyze
fileStack = listAllFiles(fullfile(basepath,imgdir));

% dimensions without resize
% resize of these variables is done automatically by the algorithm
if options.resolution == 15000
    distMin = 4;
    particleRadius = 3:5;
    th = [10:5:25];
    
    %particleRadius = 4;
    %th = 5;
    
elseif options.resolution == 20000
    distMin = 4;
    particleRadius = [3 5 7 9];
    th = [10:5:25];

elseif options.resolution == 30000
    distMin = 5;
    particleRadius = [5 7 9 11];
    th = [5:10:45];
        
elseif options.resolution == 50000
    distMin = 10;
    particleRadius = [9 11 13];
    th = [5:10:55];
end


% ------------------------------------------
switch( options.partiDetMethod )
  case {'hough'}
    th = [0];
end

% for testing purposes
% particleRadius = [10 13];
% th = [10 15];

fprintf(2,'cross val for nrun: %d..\n',options.nrun)
        
crossValLoGDetector(fileStack, particleRadius, ...
                    th, imgdir, basepath, distMin, options);


% -----------------------------------------------------------------------------------
% % % % % % Print Roc curves
% colors = 'grycmkb';
% fig = figure; title(imgdir); 
% xlabel('Precision (%)')
% ylabel('Recall (%)')
% hold on
% for radius = 1:length(particleRadius)
%     plot(ROC_precision(radius,:), ROC_recall(radius,:),'Color',colors{mod(radius,7)+1},'LineWidth',3);
%     curvesColor{radius} = ['Radius ',num2str(particleRadius(radius))];
% end
% legend(curvesColor,'Location','BestOutside')
% % % % % % Print variance for each curve
% % % % % for radius = 1:length(particleRadius)
% % % % %     plot(ROC_recall(radius,:)+ROC_varReca(radius,:),ROC_precision(radius,:)+ROC_varPrec(radius,:),'--','Color',colors{mod(radius,7)+1});
% % % % %     plot(ROC_recall(radius,:)-ROC_varReca(radius,:),ROC_precision(radius,:)-ROC_varPrec(radius,:),'--','Color',colors{mod(radius,7)+1});
% % % % % end

% filename = sprintf('%s',fullfile(basepath,imgdir,'roccurve.png'));
% print(fig,'-dpng', filename)
% figname = sprintf('%s',fullfile(basepath,imgdir,'roccurve'));
% saveas(fig, figname, 'fig')
% close all

return