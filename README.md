# Face-Recognition-Project


Requirements:
Matlab 2018a
The functions requires the Deep Learning Toolbox™ Model for ResNet-50 
Network support package and AlexNet Add-ONs. 

Valid ways to call the RecogniseFace() function:
RecogniseFace(I , 'CNN', '0')
RecogniseFace(I , 'SVM', 'ResNet')
RecogniseFace(I , 'SVM', 'HOG')
RecogniseFace(I , 'DT', 'ResNet')
RecogniseFace(I , 'DT', 'HOG')
given that I is read in the standard matlab file way. i.e. I = imread()

Valid way to call the detecNum() function:
detecNum(filename)
given that filename is a file in the current directory ending with '.jpg' or '.mp4'

The classifiers have not been uploaded on github due to their size.

Deliverables Files: 
Report.PDF
RecogniseFace.m
detectNum.m

Supportive Files used to create the functions:
Data_Base.m - code used for the creation of a database from the raw images 
videos.m - code used to extract frames from the videos 
AlexNet.m - code used to train AlexNet CNN solution
AlexNet.mat - this is the classifier   
ResNet50.m - code used to train ResNet with SVM/DT classifiers 
MDL_HOG.m - code used to train SVM/DT with HOG features 
SVM_HOG.mat - SVM classifier with HOG features 
DT_HOG.mat - DT classifier with HOG features 
SVM_ResNet50.mat - SVM classifier with Resnet features 
DT_ResNet50.mat - DT classifier with Resnet features
traimRCNN.m - script used to train the RCNN for the number detection task
DetectNum.mat - trained RCNN classifier 
imageLabelingSessionRCNN.mat - saved image labeling session
used to create the ground truth for the training of the RCNN
gTruth1.mat - ground truth for RCNN training
