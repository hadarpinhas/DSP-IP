from argparse import RawDescriptionHelpFormatter
from   pathlib              import Path
import unittest
import time
import datetime
import cv2
import numpy as np
import sys
import math
import os

class Tracker(unittest.TestCase):
    def setUp(self):
        self.startTime = time.time()

    def tearDown(self):
        t = time.time() - self.startTime
        print('%s: %.5f minutes' % (self.id(), (t/60)))

    def testOne(self):

        print('-'*100 + '\n' + ' '*50 + 'Running Tracker:' + '\n' + '-'*100)

        self.setParameters()

        self.runTracker()

    #---------------------------------------------------------------------------------------------------------------------

    def setParameters(self):

        self.basePath = Path(r"/home/yossi")
        # basePath = r"C:\Users\User\"

        # relPath = Path('Documents/database/hadar/videos/teaser/wiggle_0.mp4')
        # relPath = Path('Documents/database/hadar/videos/teaser/wiggle_0_1920_1080_AspRatio.mp4')
        relPath = Path('Documents/database/hadar/videos/police/police2.mp4')       

        pathName = relPath.stem# A/B/name.png -> name
        pathParent = relPath.parents[0] # A/B/name.png -> A/B

        curTime = datetime.datetime.now().strftime("%d%m%Y_%H%M%S")
        outRelPath = str(pathParent) + '\\' + str(pathName) + '_' + curTime + ".mp4"

        videoPath = self.basePath / relPath 
        self.outputVideoPath = os.path.join(self.basePath , outRelPath)
        print(f"{videoPath=}")
        print(f"{self.outputVideoPath=}")

        print(f"\n{videoPath=}")
        print(f"{self.outputVideoPath=}\n")

        self.fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.outputFps = 30.0

        self.cap = cv2.VideoCapture(str(videoPath))
        self.startDetectionFrame = 2400 # 

        self.waitKeyParameter = 0 # in milliseconds, -1/0 for continuous

        self.to_draw_line = False        

        self.screenSizeFactor = 1 # 0.9 for my laptop smaller screen

        self.maxNumOfFeaturesShown = 10 # number of circles annotating features (corners) found in the image

        self.initialFeatureThreshold = 0.001 # lower myShiTomasiNewFeaturesThreshold to find the first feature
        self.myShiTomasiNewFeaturesThreshold = 0.01 # Remove values < 0.01. cv2.cornerMinEigenVal outputs value=min(λ1,λ2) for each pixel.
        self.opticalFlowErrorThreshold = 10 # Removes error > Threshold. cv2.calcOpticalFlowPyrLK outputs position (p1), status (st), and error (er). 
        
        self.color = np.random.randint(0, 255, (self.maxNumOfFeaturesShown, 3))             # Create some random colors for corner annotaion

    #---------------------------------------------------------------------------------------------------------------------

    def runTracker(self):

        self.initDetector()

        self.initTracker()

        self.outputVideo = cv2.VideoWriter(filename=self.outputVideoPath, fourcc=self.fourcc, fps=self.outputFps, frameSize=(self.frameWidth,self.frameHeight))
	
        self.startTrakcerLoop()

    #---------------------------------------------------------------------------------------------------------------------

    def initDetector(self):
        # Give the configuration and weight files for the model and load the network using them.
        model_weights   = os.path.join(self.basePath, 'Documents/weights/faceDetector_yolo3/yolov3-wider_16000.weights')
        model_cfg       = os.path.join(self.basePath, 'Documents/weights/faceDetector_yolo3/yolov3-face.cfg')
        self.net = cv2.dnn.readNetFromDarknet(model_cfg, model_weights)
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)


        layers_names = self.net.getLayerNames()
        # Get the names of the output layers, i.e. the layers with unconnected outputs
        self.outputsNames = [layers_names[unconnectedLayer - 1] for unconnectedLayer in self.net.getUnconnectedOutLayers()]

        self.confThreshold = 0.1
        self.nmsThreshold = 0.5

        self.netImgWidth = 416
        self.netImgHeight = 416

    #---------------------------------------------------------------------------------------------------------------------

    def initTracker(self):

        frameNumber = 0
        success, firstFrame = self.cap.read()
        assert success == True 
        while(frameNumber < self.startDetectionFrame): # skip irrelevant frames
            _, firstFrame = self.cap.read()            
            frameNumber += 1

        if self.screenSizeFactor != 1: # for my laptop smaller screen
            firstFrame = cv2.resize(firstFrame, None, fx=self.screenSizeFactor,fy=self.screenSizeFactor)

        self.frameWidth, self.frameHeight = firstFrame.shape[1], firstFrame.shape[0]

        bbox = cv2.selectROI(firstFrame) # xi, yi, self.bboxWidth, self.bboxHeight 
        self.initBboxWidth, self.initBboxHeight = bbox[2], bbox[3]

        # Initialize CSRT Tracker
        self.tracker = cv2.TrackerCSRT_create()
        # Initialize tracker with first frame and bounding box
        ok = self.tracker.init(firstFrame, bbox)


        # self.x0, self.x1 = xi, xi + self.bboxWidth
        # self.y0, self.y1 = yi, yi + self.bboxHeight

        self.linesMask  = np.zeros_like((firstFrame))
        # self.roiMask    = np.zeros((self.frameHeight, self.frameWidth), dtype=np.uint8)
        # self.roiMask[self.y0:self.y1, self.x0:self.x1] = 255

        # self.drawFeatures(firstFrame, newFeatures = p0, oldFeatures = p0)

        cv2.imshow('frame', firstFrame) 
        
    #---------------------------------------------------------------------------------------------------------------------

    def startTrakcerLoop(self):
        
        counter = 0
        while(True):
            success, frame = self.cap.read()
            assert success == True
            counter += 1

            if self.screenSizeFactor != 1: # for my laptop smaller screen
                frame = cv2.resize(frame, None, fx=self.screenSizeFactor,fy=self.screenSizeFactor)

            # Update tracker
            ok, bbox = self.tracker.update(frame)

            if ok:# Tracking success
                print(f"Tracking success {bbox=}")

                p1 = (int(bbox[0]), int(bbox[1]))
                p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))       


                if p1[0] < 0 or p1[1] < 0 or p2[0] > frame.shape[1] or p2[1] > frame.shape[0]:
                    print("out of frame")
                    print(f"before new detection {bbox=}")
                    bbox = self.detectObject(frame)
                    print(f"after new detection {bbox=}")
                    if bbox:
                        if self.initBboxWidth > bbox[2] or self.initBboxHeight > bbox[3]:
                            print(f"out of frame and found smaller before  {bbox=}")
                            bbox = int(bbox[0]), int(bbox[1]), self.initBboxWidth, self.initBboxHeight
                            print(f"out of frame and found smaller after  {bbox=}")
                            # Initialize tracker with detected object
                            self.tracker = cv2.TrackerCSRT_create()
                            self.tracker.init(frame, bbox)
                            p1 = (int(bbox[0]), int(bbox[1]))
                            p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))


                elif self.initBboxWidth > bbox[2] or self.initBboxHeight > bbox[3]:
                    print("smaller bbox")
                    bbox = self.detectObject(frame)
                    print(f"{bbox=}")
                    if bbox:
                        # Initialize tracker with detected object
                        self.tracker = cv2.TrackerCSRT_create()
                        self.tracker.init(frame, bbox)             
                
            else:
                print("Tracking failed")
                bbox = self.detectObject(frame)
                if bbox:
                    # Initialize tracker with detected object
                    self.tracker = cv2.TrackerCSRT_create()
                    self.tracker.init(frame, bbox)                          

            if bbox:
                print(f"cv2.rectangle {bbox=}")
                print(f"{p1=}, {p2[0]-p1[0]=}, {p2[1]-p1[1]=}\n")
                cv2.rectangle(frame, p1, p2, (0, 255, 0), 1)
            cv2.imshow('frame', frame)
            print(f"{counter=}")

            self.outputVideo.write(frame)            

            if cv2.waitKey(self.waitKeyParameter) & 0xff == ord('q'): 
                break

        cv2.destroyAllWindows() 
        self.cap.release()
        self.outputVideo.release()

    #---------------------------------------------------------------------------------------------------------------------

    def detectObject(self, frame):
        # Create a 4D blob from a frame.
        blob = cv2.dnn.blobFromImage(frame, 1 / 255, (self.netImgWidth, self.netImgHeight),[0, 0, 0], 1, crop=False)

        # Sets the input to the network
        self.net.setInput(blob)

        # Runs the forward pass to get output of the output layers
        outs = self.net.forward(self.outputsNames)

        frameHeight, frameWidth = frame.shape[:2]
        confidences, bboxes = [], []

        finalMaxScorePerDetecion = 0
        # Process detections
        for out in outs: # there are 3 outs: (507,6), (2028,6), (8112,6)
            for detection in out: # detection shape = (8112, 6) = (8112, [center_x, center_y, width, height, Objectness Score, Class Scores])
                scores = detection[5:]
                classId = np.argmax(scores) # take the index of the maximal score
                if classId!=0:
                    print(f"{classId=}")
                maxScorePerDetecion = scores[classId] # confidence is the the highest score
                finalMaxScorePerDetecion = max(maxScorePerDetecion,finalMaxScorePerDetecion)

                if maxScorePerDetecion > self.confThreshold:

                    # detection[0:4] are enter_x, center_y, width, height given as ratio [0-1] from frame width/height. 
                    center_x, center_y, width, height = (detection[0:4] * np.array([frameWidth, frameHeight, frameWidth, frameHeight])).astype(int)
                    left, top = int(center_x - width / 2), int(center_y - height / 2)
                    confidences.append(float(maxScorePerDetecion))
                    bboxes.append((left, top, width, height))  # Use tuple for consistency

        print(f"{finalMaxScorePerDetecion=}")
        

        indices = cv2.dnn.NMSBoxes(bboxes=bboxes, scores=confidences, score_threshold=self.confThreshold, nms_threshold=self.nmsThreshold)

        final_boxes = [bboxes[i] for i in indices]

        bbox = []

        if len(final_boxes)==0:
            print(f"len(final_boxes)==0, {final_boxes=}")
        elif len(final_boxes)>=1:
            print(f"len(final_boxes)>=1, {final_boxes=}")
            bbox = final_boxes[0]
            bbox = max(int(bbox[0]), 0), max(int(bbox[1]), 0), bbox[2], bbox[3]
                       
        return bbox


    #---------------------------------------------------------------------------------------------------------------------

if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(Tracker)
    unittest.TextTestRunner(verbosity=0).run(suite)

