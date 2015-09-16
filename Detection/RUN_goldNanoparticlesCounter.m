function [detectedNanoparticles,annotation] = ...
    RUN_goldNanoparticlesCounter(fileStack, nanoparticleSize, th, options, distMin)
% This function allows to detect gold nanoparticle (black spots) on
% brightfield images
%
% fileStack: images to be opened and analyzed
% nanoparticleSize: size of the nanoparticles to be detected
% th: threshold value used for the nanoparticles detection

nanoparticleSize  = nanoparticleSize / options.resize;
step              = 1 / options.resize;

% close all
   

nFiles      = length(fileStack);
nAnnotators = length(options.annotators);
annotation  = cell(nFiles,nAnnotators);

% Main
for file = 1:nFiles

    imgname = fullfile(fileStack(file).path,fileStack(file).filename);
    idxnamestrip  = strfind(fileStack(file).filename,'.');
    
    % loads the annotations for each user
    annfilename = [fileStack(file).filename(1:idxnamestrip-1) ,'.csv'];
    % already performs the resize of the annotations
    annotation(file,:) = getAnnotations(options,fileStack(file).path,annfilename);
    
    fprintf(2,'Loading %s...',imgname);
    im = imread(fullfile(fileStack(file).path,fileStack(file).filename));
    im = imresize(im, 1 / options.resize);
    
    fprintf(2,'..done\n');
    % gray2rgb :)
    if length( size(im) ) < 3
        im(:,:,2) = im;
        im(:,:,3) = im(:,:,1);
    end
    
    % Obtain mask to use in the gold nanoparticles detection
    imgMask = [];
    if options.withMask
        solution = 3; % 1 - image threshold; 
                      % 2 - image entropy analysis
                      % 3 - image histogram..
        imgMask = getImgMask(im(:,:,1), distMin, solution);
    end
    
    % Gold nanoparticles Detection
    fprintf(2,'Processing..');
    switch( options.partiDetMethod )
        
      case 'log_detector'
        % LoG(im, Rmin, Rmax,step,sigma,th,disk_size,mask,signal)
        sp = LoG( im(:,:,1), nanoparticleSize-step, nanoparticleSize+step, step,...
                  1, th, floor(nanoparticleSize*0.9), ...
                  imgMask,1);
        
      case 'hough'
        nanoparticleSize
        [floor(nanoparticleSize-step), ceil(nanoparticleSize+step)]
        centers = imfindcircles(im(:,:,1), [floor(nanoparticleSize-step), ceil(nanoparticleSize+step)],...
                                'Method', 'TwoStage');
        %, 'Sensitivity', 0.75,...
        % ,'EdgeThreshold', 0.15

        % LoG Compatibility
        sp = struct('radius',[],'x',[],'y',[],'resp',[]);
        n  = 1;
        for p = 1:size(centers,1)
            sp(n).x      = round( centers(p,1) );
            sp(n).y      = round( centers(p,2) );
            n = n + 1;
        end
       
            
    end
    fprintf(2,'..Done.\n');
    
    % Remove detections and print results
    for spot = length(sp):-1:1
        % small hack to remove detected spots over the scale
        % that is always in the same location
        if ( sp(spot).x < .1*size(im,2) & ...
             sp(spot).y > .9*size(im,1) )
            sp(spot) = '';
            continue
        end
    end
    
    % gold nanoparticles
    detectedNanoparticles{file} = sp;

    if options.debug
        % for debugging purposes
        % Print manual results
        for correctSpot = 1:length(annotation{file}.xtopleft)
            I = annotation{file}.ytopleft(correctSpot)-5 : annotation{file}.ybottomright(correctSpot)+5;
            J = annotation{file}.xtopleft(correctSpot)-5 : annotation{file}.xbottomright(correctSpot)+5;
            
            ind = I < 1 | J < 1;
            I(ind) = [];
            J(ind) = [];
            im(I,J,2) = 255;
        end
        
        %figure;
        %imshow(im), hold on;
        for spdetected=1:length(sp)
            I = sp(spdetected).y-5:sp(spdetected).y+5;
            J = sp(spdetected).x-5:sp(spdetected).x+5;
            
            ind = I < 1 | J < 1;
            
            I(ind) = [];
            J(ind) = [];
            im(I,J,1) = 0;
            im(I,J,2) = 0;
            im(I,J,3) = 255;

        end
        
        % imshow(im)
        % pause
        % close all

        % pause
        % Save results
        filename    = fullfile(fileStack(file).path,'resultado',fileStack(file).filename(1:idxnamestrip-1));
        filename
        
        filenameout = sprintf('%s_%s_%03d_%03d.jpg',...
                              filename, options.partiDetMethod, nanoparticleSize, th);
        fprintf(1,'Saving file: %s\n',filenameout);
        imwrite(im,filenameout,'jpg');

    end
end