clear all 
gpuDevice(1)
doTraining = true;
if doTraining    
    % Train a network.
    % Running Our Face Detection Training 
    width = 240;
    height = 135;
    channel = 1;
    net = RobSha_FaceNet(width,height,channel);
else
    % Load pre-trained detector for the example
    load('TrainedNetwork_SAVED.mat')    
end 

TestFor = 1;
% Face Tracking Initializations Testing for Robson and Shariar
faceDetector = vision.CascadeObjectDetector();
if TestFor == 1
    videoFileReader = vision.VideoFileReader('Shahriar_Video.mov'); 
else
    videoFileReader = vision.VideoFileReader('Robson_Video.mov'); end


videoFrame      = step(videoFileReader);
bbox            = step(faceDetector, videoFrame);

% Draw the returned bounding box around the detected face.
videoOut = insertObjectAnnotation(videoFrame,'rectangle',bbox,'Face');
figure, imshow(videoOut), title('Detected face');

[hueChannel,~,~] = rgb2hsv(videoFrame);

% Display the Hue Channel data and draw the bounding boVideoFrameTest around the face.
figure, imshow(hueChannel), title('Hue channel data');
rectangle('Position',bbox(1,:),'LineWidth',2,'EdgeColor',[1 1 0])
noseDetector = vision.CascadeObjectDetector('Nose', 'UseROI', true);
nosebbox     = step(noseDetector, videoFrame, bbox(1,:));

% Create a tracker object.
tracker = vision.HistogramBasedTracker;

% Initialize the tracker histogram using the Hue channel piVideoFrameTestels from the
% nose.
initializeObject(tracker, hueChannel, nosebbox(1,:));

% Create a video player object for displaying video frames.
videoInfo    = info(videoFileReader);
videoPlayer  = vision.VideoPlayer('Position',[200 200 videoInfo.VideoSize]);

% Track the face over successive video frames until the video is finished.
while ~isDone(videoFileReader)

    % EVideoFrameTesttract the neVideoFrameTestt video frame
    videoFrame = step(videoFileReader);
    % DETECT FACE
    %bbox            = step(faceDetector, videoFrame); % 
    % RGB -> HSV
    [hueChannel,~,~] = rgb2hsv(videoFrame);
    % DETECT SHAHRIAR
    VideoFrameTest = imresize(rgb2gray(videoFrame),0.125);
    Test = classify(net,VideoFrameTest); 
    
    
    Sha = {'Shahriar'}; % Label For Shahriar 
    NotSha = {'Not-Shahriar'}; % Label For Shahriar 
    % After detecting a face we assign a name to the face based on if its
    % Shahriar or not
  
    if (Test == Sha(1))
        % Track using the Hue channel data
        bbox = step(tracker, hueChannel);
        % Insert a bounding boVideoFrameTest around the object being tracked
        videoOut = insertObjectAnnotation(videoFrame,'rectangle',bbox,'Shahriar');
         % Display the annotated video frame using the video player object
        step(videoPlayer, videoOut);end
    if (Test == NotSha(1))
        videoOut = insertObjectAnnotation(videoFrame,'rectangle',bbox,'Not-Shahriar');
         % Display the annotated video frame using the video player object
        step(videoPlayer, videoOut); end

end

% Release resources
release(videoFileReader);
release(videoPlayer);