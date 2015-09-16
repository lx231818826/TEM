function confirmMetric()
format long
addpath('../');
detected = load(['../../../imgs_nanoparticles/15000/db2/' ...
                 'resultado_sae/detectedNanoParticlesDetectionResult_log_detector_test_001.mat']);
detected = detected.detectedNanoParticles;


basepath = '../../../imgs_nanoparticles/15000/db2/';
imgdir   = '';

testIDS = [4  9  7 20  3  0 21 15 12];

options = struct();
options.annotators = {'user'};
options.resize     = 2;

annotation  = cell(length(testIDS),length(options.annotators));

% Choose files to analyze
fileStack = listAllFiles(fullfile(basepath,imgdir));

Precision_T = 0;
Recall_T    = 0;

for file=1:length(testIDS)
    
    imgname       = fullfile( fileStack(testIDS(file)+1).path, fileStack(testIDS(file)+1).filename);
    idxnamestrip  = strfind(fileStack(testIDS(file)+1).filename,'.');

    % loads the annotations for each user
    annfilename = [fileStack(testIDS(file)+1).filename(1:idxnamestrip-1) ,'.csv'];
    
    % already performs the resize of the annotations
    annotation(file,:) = getAnnotations(options,fileStack(file).path,annfilename);

    % % ---------------------------------------------------------------------------
    % b   = annotation(file,:);
    % img = imresize(imread(imgname),1/options.resize);
    % figure, imshow(img), hold on 
    % ptx = (b{1}.xbottomright + b{1}.xtopleft) / 2;
    % pty = (b{1}.ybottomright + b{1}.ytopleft) / 2;
    
    % plot(ptx,pty,'ro','MarkerSize',20)
    % x = [detected{file}.x];
    % y = [detected{file}.y];
    % for sp=1:length(detected{file})
    %     plot(x(sp),y(sp),'g+','MarkerSize',20);
    % end
    % % ---------------------------------------------------------------------------
    nAvgGoldPartiAnn = length(annotation{file,1}.xtopleft);
    
    [TP,FP,FN] = performEvaluation(detected{file},annotation(file,:),4/options.resize);
    TP
    
    Precision_T = Precision_T + (TP/(TP+FP+0.000000001)); % True positives / number of automatic detections
    Recall_T    = Recall_T  + (TP/nAvgGoldPartiAnn);    
    
    %[nAvgGoldPartiAnn, TP+FN]
    %pause
end

Precision_T = Precision_T / length(testIDS)
Recall_T = Recall_T / length(testIDS)

return