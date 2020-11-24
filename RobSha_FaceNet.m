function [net] = RobSha_FaceNet(width,height,channel)
    gpuDevice(1) 
    % Hyperparameters
    conv_num = 32;
    conv_size = 5;
    maxPool_size = 2;
    
    try
        nnet.internal.cnngpu.reluForward(1);
    catch ME
    end
    
    trainingImages = imageDatastore('Train','IncludeSubfolders',true,'LabelSource','foldernames');
    trainingLabels = trainingImages.Labels;
    size(trainingImages)

    XTrain = zeros(width,height,channel,length(trainingImages.Files),'uint8');
    for c = 1:length(trainingImages.Files)
           I = readimage(trainingImages,c);
           if (size(I,3)==3 )
               I = rgb2gray(I); end
           XTrain(:,:,channel,c) = imresize(I,[width height]);
    end

    testImages = imageDatastore('Test','IncludeSubfolders',true,'LabelSource','foldernames');
    testLabels = testImages.Labels;

    XTest = zeros(width,height,channel,length(testImages.Files),'uint8');
    for c = 1:length(testImages.Files)
           I = readimage(testImages,c);
           if (size(I,3)==3 )
               I = rgb2gray(I); end
           XTest(:,:,channel,c) = imresize(I,[width height]);
    end


    % Display a few of the training images.
    figure
    title('A few of the training images.')
    thumbnails = readall(trainingImages);
    montage(thumbnails)


    layers = [
        % conv_num x conv_num x 3 images with 'zerocenter' normalization
        imageInputLayer([width height ])
        % conv_num 5 x 5 convolutions with stride= 1 and padding = 2
        convolution2dLayer(conv_size,conv_num,'Stride',1,'Padding',2)

        %     A batch normalization layer normalizes each input channel across a mini-batch. 
        %     To speed up training of convolutional neural networks and reduce the sensitivity
        %     to network initialization, use batch normalization layers between convolutional 
        %     layers and nonlinearities, such as ReLU layers.

        batchNormalizationLayer
        % ActivationFunction ReLU
        reluLayer
        % pooling with stride= 2
        maxPooling2dLayer(maxPool_size,'Stride',2)
        % conv_num 5 x 5 convolutions with stride = 1
        convolution2dLayer(conv_size,conv_num,'Stride',1)
        batchNormalizationLayer
        % Activation Function ReLU
        reluLayer
        % max pooling with stride= 2
        maxPooling2dLayer(maxPool_size,'Stride',2)
        % convolutions with stride= 1
        convolution2dLayer(conv_size,conv_num,'Padding','same')
        % Use batch normalization layers between convolutional 
        % layers and nonlinearities, such as ReLU layers.
        batchNormalizationLayer
        % ActivationFunction ReLU
        reluLayer
        % pooling with stride = 2
        maxPooling2dLayer(maxPool_size,'Stride',2)
        % Fully Connected Layer64 
        % Activation Function ReLU
        reluLayer
        % Fully Connected Layer 4
        fullyConnectedLayer(2)
        % Activation Function Soft max
        % A Softmax function is a type of squashing function. 
        % Squashing functions limit the output of the function into the range 0 to 1.
        % This allows the output to be interpreted directly as a probability.
        softmaxLayer
        classificationLayer];

    opts = trainingOptions('sgdm', ...
        'Momentum', 0.9, ...
        'InitialLearnRate', 0.001, ...
        'LearnRateSchedule', 'piecewise', ...
        'LearnRateDropFactor', 0.1, ...
        'LearnRateDropPeriod', 8, ...
        'L2Regularization', 0.004, ...
        'MaxEpochs',10, ...
        'MiniBatchSize', 128, ...
        'Verbose', true,...
        'Plots','training-progress');
    % A trained network is loaded from disk to save time when running the
    % example. Set this flag to true to train the network.
    doTraining = true;

    if doTraining    
        % Train a network.
        net = trainNetwork(XTrain,trainingLabels, layers, opts);
    else
        % Load pre-trained detector for the example.
        load('rcnnStopSigns.mat','net')       
    end 

    % Extract the first convolutional layer weights
    w = net.Layers(2).Weights;

    % rescale the weights to the range [0, 1] for better visualization
    w = rescale(w);

    figure
    title('The first Convolutional Layer')
    montage(w)
    data = readall(testImages);


    % Run the network on the test set.
    YTest = classify(net, XTest);

    % Calculate the accuracy.
    accuracy = sum(YTest == testLabels)/numel(testLabels);

    % figure; 
    plotconfusion(YTest,testLabels)

end
