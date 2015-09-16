function [TP, FP, FN] = performEvaluation(autoResult, annotation, distMinInit)
% Compare manual detection results with the automatic ones
% Count the number of DC, FP and FN
%
% autoResult: automatic detections
% annotation: manual annotations (ground true)
% distMin: Min dist to be true detection

nAnnotators = length(annotation);

TP = 0; FP = 0; FN = 0;

for ann=1:nAnnotators
    
    % Compute centroid from the manualResult
    GT = struct('x',[],'y',[],'flag',[]);
    % GT(1) = [x, y, flag];
    % flag => 1 (detected), 0 (not detected)
    for i = 1:length(annotation{ann}.xtopleft)
        GT(i).x = double(annotation{ann}.xtopleft(i)+annotation{ann}.xbottomright(i))/2; % xx coordinate
        GT(i).y = double(annotation{ann}.ytopleft(i)+annotation{ann}.ybottomright(i))/2; % yy coordinate
        GT(i).flag = 0; % Initially they are all not detected
    end

    % Evaluate autoResult
    TPtmp = 0; FPtmp = 0;
    for detIdx = 1:length(autoResult)
        minDistIdx = [];
        distMin = distMinInit;
        % Distance to each of the manual detections
        for gtIdx = 1:length(GT)
            dist = sqrt((GT(gtIdx).x-autoResult(detIdx).x)^2 + (GT(gtIdx).y-autoResult(detIdx).y)^2);
            if dist < distMin
                distMin = dist;
                minDistIdx = gtIdx; % Save Idx
            end
        end
        % DC, FP, FN
        if ~isempty(minDistIdx) && (GT(minDistIdx).flag == 0)
            TPtmp = TPtmp + 1; % Add one true detection
            GT(minDistIdx).flag = 1; % Change flag to 1 (detected)
        else
            FPtmp = FPtmp + 1; % Add one false positive
        end
    end
    % Count the number of false negatives (spots not detected)
    FNtmp = sum([GT(:).flag] == 0);

    TP = TP + TPtmp;
    FP = FP + FPtmp;
    FN = FN + FNtmp;
end

TP = TP / nAnnotators;
FP = FP / nAnnotators;
FN = FN / nAnnotators;


return