function [images, Labels] = readMNISTdata(FName_Images, FName_Labels)
fid = fopen(FName_Images, 'rb');

A = fread(fid, 1, 'uint32');
magicNum = swapbytes(uint32(A));

A = fread(fid, 1, 'uint32');
totalImages = swapbytes(uint32(A));

A = fread(fid, 1, 'uint32');
numRows = swapbytes(uint32(A));

A = fread(fid, 1, 'uint32');
numCols = swapbytes(uint32(A));

for k = 1:totalImages
   A = fread(fid, numRows*numCols, 'uint8');
   images{k} = reshape(uint8(A), numCols, numRows)';
end
fclose(fid);
%% Подгрузка описаний (картинка на изображении)
fid = fopen(FName_Labels, 'rb');

A = fread(fid, 1, 'uint32');
magicNum = swapbytes(uint32(A));

A = fread(fid, 1, 'uint32');
totalLabels = swapbytes(uint32(A));

for k = 1:totalLabels
   A = fread(fid, 1, 'uint8');
   Labels(k) = A;
end
fclose(fid);