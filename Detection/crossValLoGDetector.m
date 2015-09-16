
%%                                                                                       
% CrossValidation for our LoG detector method
function [bestRadius, bestTh, Acc] = crossValLoGDetector(fileStack, particleRadius, th, imgdir, basepath, distMin, options)
    bestRadius = particleRadius(1);
    bestTh = th(1);
    
    bestAcc = -inf;

    % -----------------------------------------------
    % CROSS VALIDATION
    ROC_precision = zeros(length(particleRadius),length(th));
    ROC_recall    = zeros(length(particleRadius),length(th));
    ROC_varPrec   = zeros(length(particleRadius),length(th));
    ROC_varReca   = zeros(length(particleRadius),length(th));

    for radius = 1:length(particleRadius)
        for thIdx = 1:length(th)
            fprintf(2,'Radius = %03d | Th = %03d\n',particleRadius(radius),th(thIdx));
            
            detectedNanoParticles_ = {};
            Acc    = 0;
            nfolds = options.nfolds;
            for cv=1:nfolds
                train = setxor(cv,1:nfolds);
                train = train(:);
                
                trainIDX = options.val_ids(train,:);
                valIDX   = options.val_ids(cv,:);

                trainFileStack = fileStack(trainIDX);
                valFileStack   = fileStack(valIDX);

                [detectedNanoParticles,Precision_T, Recall_T] = ...
                    apply_method(valFileStack,particleRadius(radius),th(thIdx),options,distMin);
                
                ROC_precision(radius,thIdx) = ROC_precision(radius,thIdx) ...
                    + mean(Precision_T);
                ROC_varPrec(radius,thIdx)   = ROC_varPrec(radius,thIdx) ...
                    + std(Precision_T);
                ROC_recall(radius,thIdx)    = ROC_recall(radius,thIdx) ...
                    + mean(Recall_T);
                ROC_varReca(radius,thIdx)   = ROC_varReca(radius,thIdx) ...
                    + std(Recall_T);
                
                % Final Results
                mROCPrecision = mean(Precision_T);
                mROCRecall    = mean(Recall_T);
                
                Acc = Acc + 2*(mROCPrecision*mROCRecall) / (mROCPrecision+mROCRecall+eps);

                detectedNanoParticles_{cv} = detectedNanoParticles;                
            end
            
            
            detectedNanoParticles   = detectedNanoParticles_;
            detectedResultsFilename = sprintf('detectedNanoParticlesDetectionResult_%s_%s_r=%d_th=%d_%03d',...
                                              options.partiDetMethod, ...
                                              'train', ...
                                              particleRadius(radius),...
                                              th(thIdx),...
                                              options.nrun);
            % Save global data
            dataname = fullfile(basepath,imgdir,options.resultdir,detectedResultsFilename);
            save(dataname, 'detectedNanoParticles', '-v7.3')
            
            
            ROC_precision(radius,thIdx) = ROC_precision(radius,thIdx)/nfolds;
            ROC_varPrec(radius,thIdx)   = ROC_varPrec(radius,thIdx)/nfolds;
            ROC_recall(radius,thIdx)    = ROC_recall(radius,thIdx)/nfolds;
            ROC_varReca(radius,thIdx)   = ROC_varReca(radius,thIdx)/nfolds;

            % Accuracy calculation
            % Best values verification
            if Acc > bestAcc
                bestdetected = detectedNanoParticles;
                bestAcc      = Acc;
                bestTh       = th(thIdx);
                bestRadius   = particleRadius(radius);
            end
            
            fprintf(2,'Recall - %3.2f | Precision - %3.2f\n',mean(Recall_T),mean(Precision_T));
        end
    end

    % ---------------------------------------
    % results for training data
    detectedNanoParticles = bestdetected;
    detectedResultsFilename = sprintf('detectedNanoParticlesDetectionResult_%s_%s_%03d',options.partiDetMethod, 'train',options.nrun);
    % Save global data
    dataname = fullfile(basepath,imgdir,options.resultdir,detectedResultsFilename);
    save(dataname, 'detectedNanoParticles', '-v7.3')

    % results for best parametrization
    globalResultsFilename = sprintf('LoGbestParametrizationParticlesDetectionResult_%s_%s_%03d',options.partiDetMethod, 'val',options.nrun);
    % Save global data
    dataname = fullfile(basepath,imgdir,options.resultdir,globalResultsFilename);
    save(dataname, 'bestTh', 'bestRadius', '-v7.3')
    
    % results for validation data ( THESE RESULTS ARE IMPORTANT TO
    % PLOT THE PRECISION-RECALL CURVES)
    globalResultsFilename = sprintf('GlobalNanoParticlesDetectionResult_%s_%s_%03d',options.partiDetMethod, 'val',options.nrun);
    % Save global data
    dataname = fullfile(basepath,imgdir,options.resultdir,globalResultsFilename);
    save(dataname,'ROC_precision', 'ROC_varPrec', 'ROC_recall', 'ROC_varReca','-v7.3')
    
    % ---------------------------------------
    fprintf(2,'train/test..\n');
    
    testIDX  = options.test_ids;
    testFileStack  = fileStack(testIDX);

    for i=1:length(options.test_ids)
        fprintf(2,':: %d\n',options.test_ids(i));
        fprintf(2,':: %s\n',testFileStack(i).filename);
    end
    
    % -----------------------------------------------
    ROC_precision = zeros(1,1);
    ROC_recall    = zeros(1,1);
    ROC_varPrec   = zeros(1,1);
    ROC_varReca   = zeros(1,1);
    
    [detectedNanoParticles,Precision_T, Recall_T] = ...
        apply_method(testFileStack,bestRadius,bestTh,options,distMin)
    ROC_precision(1,1) = mean(Precision_T);
    ROC_varPrec(1,1)   = std(Precision_T);
    ROC_recall(1,1)    = mean(Recall_T);
    ROC_varReca(1,1)   = std(Recall_T);
    
    
    fprintf(2,':: %2.3f\n',ROC_precision(1,1))
    fprintf(2,':: %2.3f\n',ROC_recall(1,1))

    % test ids
    globalResultsFilename = sprintf('testIDS_%s_%s_%03d',options.partiDetMethod, 'test',options.nrun);
    % Save global data
    dataname = fullfile( basepath, imgdir, options.resultdir, globalResultsFilename);
    save(dataname,'testFileStack','-v7.3')
    
    % results for training data
    globalResultsFilename = sprintf('GlobalNanoParticlesDetectionResult_%s_%s_%03d',options.partiDetMethod, 'test',options.nrun);
    % Save global data
    dataname = fullfile( basepath, imgdir, options.resultdir, globalResultsFilename);
    save(dataname,'ROC_precision', 'ROC_varPrec', 'ROC_recall', 'ROC_varReca','-v7.3')

    % results for test data
    detectedResultsFilename = sprintf('detectedNanoParticlesDetectionResult_%s_%s_%03d',options.partiDetMethod, 'test',options.nrun);
    % Save global data
    dataname = fullfile( basepath, imgdir, options.resultdir, detectedResultsFilename);
    save(dataname,'detectedNanoParticles','-v7.3')

return
    
% 
    
%% -----------------------------------------------------------------------------------------------

function [detectedNanoParticles,Precision_T, Recall_T] = apply_method(fileStack,particleRadius,th,options,distMin)
    distMin = distMin / options.resize;

    % Return the detected nanoparticles and the respective annotation
    [detectedNanoParticles, annotation] = ...
        RUN_goldNanoparticlesCounter(fileStack,particleRadius,th,options,distMin);
    
    % Measure error
    TP_T = zeros(1,length(fileStack));
    FP_T = zeros(1,length(fileStack));
    FN_T = zeros(1,length(fileStack));
    Precision_T = zeros(1,length(fileStack));
    Recall_T = zeros(1,length(fileStack));
    
    for file = 1:length(fileStack)
        % get the average number of annotated gold nanoparticles
        nAnnotators = length(options.annotators);
        nAvgGoldPartiAnn = 0;
        for ann=1:nAnnotators
            nAvgGoldPartiAnn = nAvgGoldPartiAnn + length(annotation{file,ann}.xtopleft);
        end
        nAvgGoldPartiAnn = nAvgGoldPartiAnn / nAnnotators;
        
        % Compare automatic detections with manual detections
        [TP, FP, FN] = performEvaluation(detectedNanoParticles{file}, annotation(file,:),distMin);
        TP_T(file) = TP;
        FP_T(file) = FP;
        FN_T(file) = FN;
        Precision_T(file) = (TP/(TP+FP+0.000000001)); % True positives / number of automatic detections
        Recall_T(file)    = (TP/nAvgGoldPartiAnn); % True positives / number of existing gold nanoparticles
    
    
        fprintf(2,':: Recall: %2.3f\n',Recall_T(file));
        fprintf(2,':: Precision: %2.3f\n',Precision_T(file));
    end

    return