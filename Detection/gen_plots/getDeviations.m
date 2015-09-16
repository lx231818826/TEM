

function getDeviations()

% path
path = fullfile('..','..','imgs');

% imgs
imgs = {'15000','20000','30000','50000'};

% methods
mmethods = {'icy','log_detector'};

for metIDX=1:length(mmethods)
    method = mmethods{metIDX};
    for imIDX=1:length(imgs)
        filepath = fullfile(path,imgs{imIDX},'db1','resultado');
        
        filenamepath = sprintf('%s_%s.mat',...
                               fullfile(filepath,'GlobalNanoParticlesDetectionResult'),...
                               method)
        fprintf(1,'loading %s\n',filenamepath);
        f = load(filenamepath);

        switch( method )
          case 'icy'
            %f.ROC_varPrec(1,:,:)
            %f.ROC_varReca(1,:,:)
        
            %pause
          case 'log_detector'
            
            f.ROC_varPrec
            pause 
        end
        
    end

end


return



