clear;close;clc
%% '================ Written by Farhad AbedinZadeh ================'
%                                                                  %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
%% Dataset
% datasets = '.\Covid19-Xray\Data-2classes';
% datasets = '.\Covid19-Xray\Data-3classes';

%% Create image datastore and read labels of sub-folders
exts = {'.jpg','.png','.tif'};
imds = imageDatastore(datasets, ...
    'IncludeSubfolders',true, ...
    'LabelSource','foldernames','FileExtensions',exts); 

%% Number of runs
for i =1:3 % number of runs
    
% Load pretrained network
net = alexnet;
% net = googlenet;
% net = squeezenet;
%--------------------

%% Train and Validation    
% Randomly divide the data into training and validation data sets. 
% Use 90% of the images for training and 10% for validation.

[imdsTrain,imdsValidation] = splitEachLabel(imds,0.9,'randomize');

%% Analyze Network
% Display an interactive visualization of the network architecture and 
% detailed information about the network layers.

% analyzeNetwork(net)

% The first element of the Layers property of the network is the image input layer.
inputSize = net.Layers(1).InputSize; 

%% prepare for replace
% Extract the layer graph from the trained network. 
% If the network is a SeriesNetwork object, such as AlexNet
% then convert the list of layers in net.Layers to a layer graph.

if isa(net,'SeriesNetwork') 
  lgraph = layerGraph(net.Layers); 
else
  lgraph = layerGraph(net);
end 

% Find the names of the two layers to replace
 [learnableLayer,classLayer] = findLayersToReplace(lgraph);
  
 
numClasses = numel(categories(imdsTrain.Labels));

if isa(learnableLayer,'nnet.cnn.layer.FullyConnectedLayer')
    newLearnableLayer = fullyConnectedLayer(numClasses, ...
        'Name','new_fc', ...
        'WeightLearnRateFactor',10, ...
        'BiasLearnRateFactor',10);
    
elseif isa(learnableLayer,'nnet.cnn.layer.Convolution2DLayer')
    newLearnableLayer = convolution2dLayer(1,numClasses, ...
        'Name','new_conv', ...
        'WeightLearnRateFactor',10, ...
        'BiasLearnRateFactor',10);
end

lgraph = replaceLayer(lgraph,learnableLayer.Name,newLearnableLayer);


% The classification layer specifies the output classes of the network. 
% Replace the classification layer with a new one without class labels. 
% trainNetwork automatically sets the output classes of the layer at training time. 

newClassLayer = classificationLayer('Name','new_classoutput');
lgraph = replaceLayer(lgraph,classLayer.Name,newClassLayer);


% Train Network

augimdsTrain = augmentedImageDatastore(inputSize(1:2),imdsTrain,...
    'ColorPreprocessing','gray2rgb'); % NO augmentation

%% Image Augmentation
% To automatically resize the validation images without performing further data augmentation, 
% use an augmented image datastore without specifying any additional preprocessing operations.

augimdsValidation = augmentedImageDatastore(inputSize(1:2),imdsValidation,...
    'ColorPreprocessing','gray2rgb');

%% Train options
% Specify training options

miniBatchSize = 10;
valFrequency = floor(numel(augimdsTrain.Files)/miniBatchSize);
options = trainingOptions('sgdm', ...
    'MiniBatchSize',miniBatchSize, ...
    'MaxEpochs',10, ...
    'InitialLearnRate',3e-4, ...
    'Shuffle','every-epoch', ...
    'ValidationData',augimdsValidation, ...
    'ValidationFrequency',valFrequency, ...
    'Verbose',false, ...
    'Plots','training-progress');

net = trainNetwork(augimdsTrain,lgraph,options);

%% Classify & Metrics
% Classify the validation images using the fine-tuned network, 
% and calculate the classification metrics.

[YPred,probs] = classify(net,augimdsValidation);
% accuracy = mean(YPred == imdsValidation.Labels);


% Get the known labels
testLabels = imdsValidation.Labels;

% Tabulate the results using a confusion matrix.
confMat = confusionmat(testLabels, YPred);
display(confMat)

accuracy = (sum(diag(confMat))/sum(sum(confMat)))*100;

% The metric names below depend on the sub-folder labels
% please check and change metric names accordingly

if numClasses == 2 
sensitivity = (confMat(1,1)/(confMat(1,1)+confMat(1,2)))*100; % COVID
specificity = (confMat(2,2)/(confMat(2,1)+confMat(2,2)))*100; % NORMAL
precision = (confMat(1,1)/(confMat(1,1)+confMat(2,1)))*100; 
f1score = (2*confMat(1,1))/(2*confMat(1,1)+confMat(1,2)+confMat(2,1));

elseif numClasses == 3

sensitivity = (confMat(1,1)/(confMat(1,1)+confMat(1,2)+confMat(1,3)))*100; % COVID
specificity = ((confMat(2,2)+confMat(3,3))/...
    (confMat(2,2)+confMat(3,3)+confMat(2,1)+confMat(2,3)+...
     confMat(3,1)+confMat(3,2)))*100; % NON-COVID
precision = (confMat(1,1)/(confMat(1,1)+confMat(2,1)+confMat(3,1)))*100; 
f1score = (2*confMat(1,1))/(2*confMat(1,1)+confMat(2,1)+confMat(3,1)+...
           confMat(1,2)+confMat(1,3));
    
end

% Comput AUC
cgt = double(testLabels);
cscores = double(probs);
[X,Y,T,area,OPTROCPT,SUBY,SUBYNAMES] = perfcurve(cgt,cscores(:,1),1);

%-----------------------------    
ACC=[ACC,accuracy];
SEN=[SEN,sensitivity];
SPE=[SPE,specificity];
PRE=[PRE,precision];
F1 = [F1,f1score];
AUC = [AUC,area];

end 

%% Mean & Standard Deviation
meanACC = mean(ACC);
meanSEN = mean(SEN);
meanSPE = mean(SPE);
meanPRE = mean(PRE);
meanF1 = mean(F1);
meanAUC = mean(AUC);


stdACC = std(ACC);
stdSEN = std(SEN);
stdSPE = std(SPE);
stdPRE = std(PRE);
stdF1 = std(F1);
stdAUC = std(AUC);


