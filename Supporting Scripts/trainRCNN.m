clear all
close all
clc

%% 
% digitDatasetPath = fullfile(matlabroot,'toolbox','nnet','nndemos', ...
%     'nndatasets','DigitDataset');
% imds = imageDatastore(digitDatasetPath, ...
%     'IncludeSubfolders',true,'LabelSource','foldernames');
% 
% %%
% figure;
% perm = randperm(10000,20);
% for i = 1:20
%     subplot(4,5,i);
%     imshow(imds.Files{perm(i)});
% end

%% 
% Load pretrained ResNet-50.
net = resnet50();

% Convert network into a layer graph object to manipulate the layers.
lgraph = layerGraph(net);

% figure
% plot(lgraph)
% ylim([-5 16])

% Remove the the last 3 layers. 
layersToRemove = {
    'fc1000'
    'fc1000_softmax'
    'ClassificationLayer_fc1000'
    };

lgraph = removeLayers(lgraph, layersToRemove);

%% Display the results after removing the layers.
figure
plot(lgraph)
ylim([-5 16])


%% Add the new layers
numClassesPlusBackground = 1 + 1;

% Define new classfication layers
newLayers = [
    fullyConnectedLayer(numClassesPlusBackground, 'Name', 'rcnnFC')
    softmaxLayer('Name', 'rcnnSoftmax')
    classificationLayer('Name', 'rcnnClassification')
    ];

% Add new layers
lgraph = addLayers(lgraph, newLayers);

% Connect the new layers to the network. 
lgraph = connectLayers(lgraph, 'avg_pool', 'rcnnFC');

% Display the final R-CNN network. This can be trained using trainRCNNObjectDetector.
figure
plot(lgraph)
ylim([-5 16])

%%  Train a Detector with RCNN 

options = trainingOptions('sgdm', ...
  'MiniBatchSize', 32, ...
  'InitialLearnRate', 1e-6, ...
  'MaxEpochs', 10);

rcnn = trainRCNNObjectDetector(gTruth, lgraph, options, 'NegativeOverlapRange', [0 0.3]);


%% Test the Network 
img = imread('test2.jpg');

[bbox, score, label] = detect(rcnn, img, 'MiniBatchSize', 32);

[score, idx] = max(score);

bbox = bbox(idx, :);
annotation = sprintf('%s: (Confidence = %f)', label(idx), score);

detectedImg = insertObjectAnnotation(img, 'rectangle', bbox, annotation);

figure;
imshow(detectedImg)


%% OCR
NumImage = imcrop(img,bbox(1,:));

imshow(NumImage);

results = ocr(NumImage, 'CharacterSet','0123456789','TextLayout','Block');

results.Text

res = str2num(result.Text)

%% Video
video = vision.VideoFileReader('IMG_20190128_201734.mp4');
viewer = vision.VideoPlayer;
while ~isDone(video)
    image = step(video);
    [bboxes, scores] = step(peopleDetector,image); % croppped image 
    I_people = insertObjectAnnotation(image,'rectangle',bboxes,scores);
    step(viewer,I_people);
end