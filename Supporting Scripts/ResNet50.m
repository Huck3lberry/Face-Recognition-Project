clc
clear all 
close all 

%% Load data and show data 

imds = imageDatastore('data4', 'LabelSource', 'foldernames', 'IncludeSubfolders',true);
pic = find(imds.Labels == '1', 1);

figure
imshow(readimage(imds,pic))

%% Load pretrained network
net = resnet50();

% Visualize net 
figure
plot(net)
title('First section of ResNet-50')
set(gca,'YLim',[150 170]);

% Split test and train sets 
[trainingSet, testSet] = splitEachLabel(imds, 0.6, 'randomize');

%% Create augmentedImageDatastore from training and test sets to resize
% images in imds to the size required by the network.
imageSize = net.Layers(1).InputSize;

imageAugmenter = imageDataAugmenter( ...
    'RandRotation',[-20,20], ...
    'RandXTranslation',[-3 3], ...
    'RandYTranslation',[-3 3]);

augmentedTrainingSet = augmentedImageDatastore(imageSize, trainingSet,'DataAugmentation',imageAugmenter,'ColorPreprocessing', 'gray2rgb' );
augmentedTestSet = augmentedImageDatastore(imageSize, testSet, 'DataAugmentation',imageAugmenter,'ColorPreprocessing', 'gray2rgb');

%% Get the network weights for the second convolutional layer
w1 = net.Layers(2).Weights;

% Scale and resize the weights for visualization
w1 = mat2gray(w1);
w1 = imresize(w1,5); 

% Display a montage of network weights. There are 96 individual sets of
% weights in the first layer.
figure
montage(w1)
title('First convolutional layer weights')


featureLayer = 'fc1000';
trainingFeatures = activations(net, augmentedTrainingSet, featureLayer, ...
    'MiniBatchSize', 32, 'OutputAs', 'columns');

%% Train A Multiclass SVM and DT Classifiers Using CNN Features
% Get training labels from the trainingSet
trainingLabels = trainingSet.Labels;

% Train multiclass SVM classifier using a fast linear solver, and set
% 'ObservationsIn' to 'columns' to match the arrangement used for training
% features.
%classifier = fitcecoc(trainingFeatures, trainingLabels, ...
 %   'Learners', 'svm', 'Coding', 'onevsall', 'ObservationsIn', 'columns');

% Uncomment to Train A Decision Tree classifier 
classifier = fitcecoc(trainingFeatures, trainingLabels, ...
    'Learners', 'tree', 'Coding', 'onevsall', 'ObservationsIn', 'columns');
%% Evaluate Classifier 
% Extract test features using the CNN
testFeatures = activations(net, augmentedTestSet, featureLayer, ...
    'MiniBatchSize', 32, 'OutputAs', 'columns');

% Pass CNN image features to trained classifier
predictedLabels = predict(classifier, testFeatures, 'ObservationsIn', 'columns');

% Get the known labels
testLabels = testSet.Labels;


figure;
% Tabulate the results using a confusion matrix.
confMat = confusionmat(testLabels, predictedLabels);
confusionchart(confMat);
% Convert confusion matrix into percentage form
confMat = bsxfun(@rdivide,confMat,sum(confMat,2))

% Display the mean accuracy
Accuracy = mean(diag(confMat))

%% Apply the trained classifier on the Test Image 
testImage = readimage(testSet,1);
testLabel = testSet.Labels(1)

% Create augmentedImageDatastore to automatically resize the image when
% image features are extracted using activations.
ds = augmentedImageDatastore(imageSize, testImage, 'ColorPreprocessing', 'gray2rgb');

% Extract image features using the CNN
imageFeatures = activations(net, ds, featureLayer, 'OutputAs', 'columns');

% Make a prediction using the classifier
predictedLabel = predict(classifier, imageFeatures, 'ObservationsIn', 'columns')
