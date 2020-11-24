a=VideoReader('test5.mov');
for img = 1:a.NumberOfFrames;
    filename=strcat('frame',num2str(img),'.jpg');
    b = read(a, img);
    imshow(b);
    imwrite(b,filename);
end