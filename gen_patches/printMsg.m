function printMsg(msg,debug)

if debug > 0
    fprintf(1,msg);
else
    fprintf(1,'.');
end

return