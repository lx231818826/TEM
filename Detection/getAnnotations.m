
%%                                                                 
function annotation = getAnnotations(options,path,annfilename)

nAnnotators = length(options.annotators);
annotation  = cell(1,nAnnotators);

for ann=1:nAnnotators
    annfilenameUser = fullfile(path,'annotation/',options.annotators{ann},annfilename);
    fprintf(1,'Annotation file %s\n',annfilenameUser)
    annotation{ann} = mycsvreader(annfilenameUser,options);
end

return