clc
clear all
close all 
% Extracting Frames from Videos 

obj = VideoReader('IMG_3369.mp4');
%obj = mmreader('IMG_20190128_201734.mp4');
vid = read(obj);
frames = obj.NumberOfFrames;
for x = 1 : 85
    imwrite(vid(:,:,:,x),strcat('frame-',num2str(x),'.png'));
end

