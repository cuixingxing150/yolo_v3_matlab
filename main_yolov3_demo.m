%% params
root_path = './model';
cfg_path = fullfile(root_path,'yolov3-tiny.cfg');
bin_path = fullfile(root_path,'yolov3-tiny.weights');
class_path = fullfile(root_path,'coco.names');
confThreshold = 0.1;
nmsThreshold = 0.1;

%% init must only once
YoloV3Detect('init',cfg_path,bin_path,class_path,confThreshold,nmsThreshold)

%% detect can run more than once
img = imread('person.jpg');
tic;
[prredictROIs,prredictScores,prredictLabels] = YoloV3Detect('detect',img);
toc
RGB = insertObjectAnnotation(img,'rectangle',prredictROIs,prredictLabels);
imshow(RGB)
