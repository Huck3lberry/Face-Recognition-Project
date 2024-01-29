clc 
clear all 
close all 

%% Load 
faceDatabase = imageSet('data4','recursive');

imshow(read(faceDatabase(1),1)); 
%2 is the column of the data matrix(person) and 4 is the image in the
%folder 

%% %% Face Detection

% Display Query Image and Database Side-Side
personToQuery = 1;
galleryImage = read(faceDatabase(personToQuery),1);
figure;
for i=1:size(faceDatabase,2)
    imageList(i) = faceDatabase(i).ImageLocation(1);
end
subplot(1,2,1);imshow(galleryImage);
subplot(1,2,2);montage(imageList);


%% Split Database into Training & Test Sets
[training,test] = partition(faceDatabase,[0.6 0.4]);

% Extract and display Histogram of Oriented Gradient Features for single face
person = 1;
[hogFeature, visualization]= ...
extractHOGFeatures(read(training(person),1));
figure;
subplot(2,1,1);imshow(read(training(person),1));title('Input Face');
subplot(2,1,2);plot(visualization);title('HoG Feature');

%% Extract HOG Features for training set
trainingFeatures = zeros(size(training,2)*training(1).Count,4680);
featureCount = 1;
for i=1:size(training,2)
    for j = 1:training(i).Count
        trainingFeatures(featureCount,:) = extractHOGFeatures(read(training(i),j));
        trainingLabel{featureCount} = training(i).Description;
        featureCount = featureCount + 1;
    end
personIndex{i} = training(i).Description;
end

%% Create class classifier 
faceClassifier = fitcecoc(trainingFeatures,trainingLabel);

% Uncomment the row below to train a Decison Tree Classifier 
%faceClassifier = fitctree(trainingFeatures,trainingLabel, 'Prune','on');
%% Use classifier to predict 
person = 1;

queryImage = read(test(person),1);
queryFeatures = extractHOGFeatures(queryImage);
personLabel = predict(faceClassifier,queryFeatures);

%% Extract HOG Features for TEST set
testFeatures = zeros(size(test,2)*test(1).Count,size(hogFeature,2));
featureCounttest = 1;
for i=1:size(test,2)
    for j = 1:test(i).Count
        testFeatures(featureCounttest,:) = extractHOGFeatures(read(test(i),j));
        testLabel{featureCounttest} = test(i).Description;
        featureCounttest = featureCounttest + 1;
    end
personIndex{i} = test(i).Description;
end
testLabel12 = predict(faceClassifier,testFeatures);
%% Map back to training set to find identity
booleanIndex = strcmp(personLabel, personIndex);
integerIndex = find(booleanIndex);
subplot(1,2,1);imshow(queryImage);title('Query Face');
subplot(1,2,2);imshow(read(training(integerIndex),1));title('Matched Class');

%% Test First 5 People from Test Set
%Now we can use the following loop to test our classifier on a few more images:
figure;
figureNum = 1;
for person=1:5
    for j = 1:test(person).Count
        queryImage = read(test(person),j);
        queryFeatures = extractHOGFeatures(queryImage);
        personLabel = predict(faceClassifier,queryFeatures);
        % Map back to training set to find identity
        booleanIndex = strcmp(personLabel, personIndex);
        integerIndex = find(booleanIndex);
        subplot(4,2,figureNum);imshow(imresize(queryImage,3));title('Query Face');
        subplot(4,2,figureNum+1);imshow(imresize(read(training(integerIndex),1),3));title('MatchedClass');
        figureNum = figureNum+2;
    end
figure;
figureNum = 1;
end
%% Performance Metrics 
testLabel12 = string(testLabel12);
testLabel12 = double(testLabel12);
testLabel = string(testLabel);
testLabel = double(testLabel);
testLabel = testLabel'
figure;
C =confusionmat(testLabel12, testLabel);
confusionchart(C);
%plotconfusion(testLabel12', testLabel);
Accuracy = mean(diag(C)/4)

