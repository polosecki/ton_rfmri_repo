function f = file_exists(dd,subj,run)

f =0;
for TR_id = 1:dd.TRs
    
    [infile,err]= sprintf(dd.sprintf_format,dd.mypath,dd.subjects(subj),dd.subjects(subj),dd.task,TR_id);
    fin=fopen(infile,'r');
    if fin < 0
        display 'error opening file';
        display infile;
        return;
    else
        f=1;
        fclose(fin);
        return
    end

end