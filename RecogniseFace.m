function P = RecogniseFace(I, classifier, featureType)
    %Detect faces 
    FaceDetector = vision.CascadeObjectDetector();
    BBOX=step(FaceDetector,I);
    n=size(BBOX,1); % number of faces 
    if n == 0 % if number of faces is 0
        P = [];
        
    else  % if faces are present ==>
        
        % Iterate through the detected face to classify them
        for k = 1:n  
    
                 x = (BBOX(k,1)+BBOX(k,3)+BBOX(k,1))/2;
                 y = (BBOX(k,2)+BBOX(k,4)+BBOX(k,2))/2;
                 y = categorical(y);
                 x = categorical(x);
            

%              
                 a = BBOX(k,1);
                 b = BBOX(k,2);
                 c = a+BBOX(k,3);
                 d = b+BBOX(k,4);

                 faceImage= I(b:d,a:c,:);                            
                 faceImage = imresize(faceImage,[112 92]);

               % Classify the face for a chosen classification method  
                 
            if strcmp(classifier, 'AlexNet') & strcmp(featureType, '0'); 
                net = alexnet;
                inputSize = net.Layers(1).InputSize;

                augimdsValidation = augmentedImageDatastore(inputSize(1:2),faceImage, 'ColorPreprocessing', 'gray2rgb');
                load netTransfer1.mat

                [YPred,scores] = classify(netTransfer,augimdsValidation);
                id = YPred;
                
                P(k,:) = [id,x,y];
         
         
            elseif strcmp(classifier, 'DT') & strcmp(featureType, 'ResNet');
                            
                    load DT_ResNet1_Classifier.mat 

                % Create augmentedImageDatastore to automatically resize the image when
                % image features are extracted using activations.
                    net = resnet50();
                    imageSize = net.Layers(1).InputSize;
                    ds = augmentedImageDatastore(imageSize, faceImage, 'ColorPreprocessing', 'gray2rgb');

                % Extract image features using the CNN
                    featureLayer = 'fc1000';
                    imageFeatures = activations(net, ds, featureLayer, 'OutputAs', 'columns');

                % Make a prediction using the classifier
                    id = predict(classifier, imageFeatures, 'ObservationsIn', 'columns');


                    P(k,:) = [id,x,y];
                    
                    
                    
                    
            elseif strcmp(classifier, 'SVM') & strcmp(featureType, 'ResNet');
                  load SVM_ResNet1_Classifier.mat 

            % Create augmentedImageDatastore to automatically resize the image when
            % image features are extracted using activations.
                net = resnet50();
                imageSize = net.Layers(1).InputSize;
                ds = augmentedImageDatastore(imageSize, faceImage, 'ColorPreprocessing', 'gray2rgb');

            % Extract image features using the ResNET
                featureLayer = 'fc1000';
                imageFeatures = activations(net, ds, featureLayer, 'OutputAs', 'columns');

            % Make a prediction using the pretrained classifier
                id = predict(classifier, imageFeatures, 'ObservationsIn', 'columns');
                P(k,:) = [id,x,y];
            
                
                
            
            elseif  strcmp(classifier, 'DT') & strcmp(featureType, 'HOG');
  
            % Load the classifier 
                load DT_HOG1_Classifier.mat 
            
            % Use classifier to predict           
                queryFeatures = extractHOGFeatures(faceImage);
                id = predict(faceClassifier,queryFeatures);
                
                P(k,:) = [id,x,y];
                
                
                
                
                
                
            else  strcmp(classifier, 'SVM') & strcmp(featureType, 'HOG');

                % Load the classifier 
                load SVM_HOG1_Classifier.mat 

                % Use classifier to predict           
                queryFeatures = extractHOGFeatures(faceImage);
                id = predict(faceClassifier,queryFeatures);
                P(k,:) = [id,x,y];
                
                
                
                
            
                
            end
            
            
        end
    end
end

