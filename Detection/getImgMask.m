function imgMask = getImgMask(img, distMin, solution)

%% Solution 1
if solution == 1
    % Get minimum value
    minValue = min(min(img)) + 1; % Add 1 in case of min = 0;
    
    % Creat mask
    imgMask = img < minValue*2; % Multiply by 2 just to make sure that the mask
                                % includes all the nanoparticles
    imgMask = bwmorph(imgMask,'dilate',distMin);
    
elseif solution == 2
    %% Solution 2
    
    % Measure image entropy according to the nanoparticles size
    nhood = strel('disk', round(distMin/2));
    entropyMask = entropyfilt(uint8(img),nhood.getnhood);
    mask = entropyMask./max(max(entropyMask));
    meanth = mean(mean(mask));
    
    % Creat mask
    imgMask = mask < meanth/2;
    
    % Dilate to make sure that the mask includes all the nanoparticles
    imgMask = bwmorph(imgMask,'dilate',distMin);

elseif solution == 3
    % Threshold image
    % [labels, region] = MSER(img); % Is taking to long to obtain the results
    minTh = 1;
    step  = 2;
    maxTh = 125;
    multipleTh = zeros(size(img));
    for th = minTh:step:maxTh
        % Image partition
        imbw = img < th;
        
        % Add previous result to the multipleTh matrix
        multipleTh = multipleTh + imbw;
    end
    % Get local maxima from multipleTh (high frequency of pixels given multiple th values)
    %imgMask = imregionalmax(multipleTh);
    %whos imgMask
    maxv = max(multipleTh(:));
    imgMask = img < (maxv+20); 

    %figure, imshow(imgMask)
    %akak
    %fprintf(2,'%d\n',size(imgMask,1));

    % Dilate to make sure that the mask includes all the nanoparticles
    imgMask = bwmorph(imgMask,'dilate',distMin);

    % % Calculate average size of regions (nanoparticles)
    % imgProps = regionprops(roi,'Area');
    % %rArea = [imgProps.Area];
    % %rArea(rArea == max(rArea)) = []; % remove background area
    % %rArea(rArea < 3) = []; % remove dots
    % %averageSize = mean(rArea);

    % % Get centroid of each region
    % imgProps = regionprops(roi,'Centroid','EquivDiameter','Area');

    % n = 0;
    % sp = struct('radius',[],'x',[],'y',[],'resp',[]);
    % for i = 1:length(imgProps)
    %     if imgProps(i).Area < distMin % > averageSize
    %         n = n + 1;
    %         sp(n).x      = imgProps(i).Centroid(1);
    %         sp(n).y      = imgProps(i).Centroid(2);
    %         sp(n).radius = imgProps(i).EquivDiameter/2;
    %     end
    % end
end