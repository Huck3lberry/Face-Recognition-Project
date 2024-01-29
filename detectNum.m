function [num] = detectNum(filename)
%detectNum() accepts an image file - jpeg,jpg etc 
[~,~,ext] = fileparts(filename);

    % OCR in case of image file jpg
    if ext == '.jpg'
        filename = imread(filename);    
        
        load DetectNum.mat

        [bbox, score, label] = detect(rcnn, filename, 'MiniBatchSize', 32);

        [score, idx] = max(score);

        bbox = bbox(idx, :);
        annotation = sprintf('%s: (Confidence = %f)', label(idx), score);

        detectedImg = insertObjectAnnotation(filename, 'rectangle', bbox, annotation);

        NumImage = imcrop(filename,bbox(1,:));

        results = ocr(NumImage, 'CharacterSet','0123456789','TextLayout','Block');

        num = results.Text;
        
    end
        
        
        % OCR in case of a video file 
    if ext == '.mp4'
        videoreader = VideoReader(filename);
        filename = read(videoreader,1);
          load DetectNum.mat

        [bbox, score, label] = detect(rcnn, filename, 'MiniBatchSize', 32);

        [score, idx] = max(score);

        bbox = bbox(idx, :);
        annotation = sprintf('%s: (Confidence = %f)', label(idx), score);

        detectedImg = insertObjectAnnotation(filename, 'rectangle', bbox, annotation);

        NumImage = imcrop(filename,bbox(1,:));

        results = ocr(NumImage, 'CharacterSet','0123456789','TextLayout','Block');

        num = results.Text;
        
        
        
    end
        
        
end

