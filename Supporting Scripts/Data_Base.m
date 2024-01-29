clc
close all
clear all 

for k=1:21
   % Specify the data folder with images path  
   basepath1 = 'C:/Users/Kiril/Desktop/City University/Computer Vision/CW/data1/57/%d.png';
   path = sprintf(basepath1,k);
   
   a=imread(path);
   disp(size(a));
   % a = imrotate(a, -90,'bilinear', 'crop'); Uncomment to fix the rotation  
   A=imresize(a,[900,600]);
   
   % Eyepair detection  
   FaceDetector = vision.CascadeObjectDetector();
   BBOX=step(FaceDetector,A);
   B = insertObjectAnnotation(A,'rectangle',BBOX,'Face');
   imshow(B), title('detected Faces');
   n=size(BBOX,1);
   str_n=num2str(n);
   str = strcat('no of detected faces are =', str_n);
   disp(str);
   % Detecting and croping the detected faces by eyes and nose/mouth
   for i=1:n
       faceImage = imcrop(A,BBOX(1,:));
       figure; imshow(faceImage);
       
       %FaceDetector1 = vision.CascadeObjectDetector('EyePairBig');
       FaceDetector1 = vision.CascadeObjectDetector('Nose');
       BBOX1 = step(FaceDetector1,faceImage);
       B1 = insertObjectAnnotation(faceImage,'rectangle',BBOX1,'Face');
       figure, imshow(B1), title('detected eyes');
       n1 = size(BBOX1,1);
       str_n1 = num2str(n1);
       str1 = strcat('no of detected noses are = ', str_n1);
       disp(str1);
       if n1>0
           J = imresize(faceImage,[112 92]);
           l = rgb2gray(J);
           figure, imshow(l);
           
           basepath = 'C:/Users/Kiril/Desktop/City University/Computer Vision/CW/data2/57/%d.png';
           path = sprintf(basepath,k);
           imwrite(l,path); % saves the cropped face in a folder
       end 
   end
    % Detecting and croping the detected faces by eyes and nose/mouth
   for i=1:n
       faceImage1 = imcrop(A,BBOX(1,:));
       figure; imshow(faceImage1);
       
       FaceDetector2 = vision.CascadeObjectDetector('Mouth');
       BBOX2 = step(FaceDetector2,faceImage1);
       B2 = insertObjectAnnotation(faceImage1,'rectangle',BBOX2,'Face');
       figure, imshow(B2), title('detected eyes');
       n2 = size(BBOX2,1);
       str_n2 = num2str(n2);
       str2 = strcat('no of detected mouths are = ', str_n2);
       disp(str2);
       if n2>0
           L = imresize(faceImage,[112 92]);
           E = rgb2gray(L);
           figure, imshow(E);
           
           basepath = 'C:/Users/Kiril/Desktop/City University/Computer Vision/CW/data2/57/%d.png';
           path = sprintf(basepath,k);
           imwrite(E,path); % saves the cropped face in a folder 
       end 
   end
     
    
end