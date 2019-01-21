function exists = exists_file(infile)

% just checs if the file exists
fin=fopen(infile,'r');
if fin < 0
    exists = 0;
else
    exists = 1;
    fclose(fin);
end
