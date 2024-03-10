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

    def getNewFeatures(self, newFrame, Threshold=None):
        if Threshold == None:
            Threshold = self.myShiTomasiNewFeaturesThreshold
        # This function returns all features with threshold > absolute val
        src = newFrame[self.y0: self.y1, self.x0: self.x1]
        src_gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
        cv2.imshow('src_gray', src_gray) 

        myShiTomasi_dst = cv2.cornerMinEigenVal(src=src_gray, blockSize=3, ksize=3, borderType=cv2.BORDER_REFLECT_101)

        goodFeatures = []
        myShiTomasi_minVal, myShiTomasi_maxVal, _, _ = cv2.minMaxLoc(myShiTomasi_dst)

        for i in range(src_gray.shape[0]):
            for j in range(src_gray.shape[1]):
                if myShiTomasi_dst[i,j] > Threshold: # myShiTomasi_minVal + ( myShiTomasi_maxVal - myShiTomasi_minVal ) * self.myShiTomasiThreshold:
                    goodFeatures.append([[self.x0 + j, self.y0 + i]])

        return goodFeatures

    #---------------------------------------------------------------------------------------------------------------------

    def shiftRoiMask(self, p1):

        p1X, p1Y = p1[0].ravel().astype(np.uint32)

        self.x0, self.x1 = p1X - self.bboxWidth //2, p1X + self.bboxWidth //2
        self.y0, self.y1 = p1Y - self.bboxHeight//2, p1Y + self.bboxHeight//2
        print(f"{p1=}")
        print(f"{self.frameWidth=}")
        print(f"{self.frameHeight=}")

        if self.x0 < 0 or self.x1 > self.frameWidth or self.y0 < 0 or self.y1 > self.frameHeight:
            print(self.x0 < 0 or self.x1 > self.frameWidth)
            print(self.y0 < 0 or self.y1 > self.frameHeight)
            sys.exit("out of screen!")

        self.roiMask    = np.zeros((self.frameHeight, self.frameWidth), dtype=np.uint8)
        self.roiMask[self.y0:self.y1, self.x0:self.x1] = 255

    #---------------------------------------------------------------------------------------------------------------------

    def getOpticalFlowFeatures(self, oldImg, newImg, p0):
        # Calculates an optical flow for a sparse feature set using the iterative Lucas-Kanade method with pyramids.            
        # Assuming:         
        # 1.The pixel intensities of an object do not change between consecutive frames.
        # 2. Neighbouring pixels have similar motion.

        oldGray = cv2.cvtColor(oldImg, cv2.COLOR_BGR2GRAY)
        newGray = cv2.cvtColor(newImg, cv2.COLOR_BGR2GRAY)

        p1, st, err = cv2.calcOpticalFlowPyrLK(
                prevImg=oldGray,        # first 8-bit input image
                nextImg=newGray,        # second input image
                prevPts=p0,             # vector of 2D points for which the flow needs to be found.
                nextPts=None,
                winSize = (21, 21),       # size of the search window at each pyramid level
                maxLevel = 2,           #0-based maximal pyramid level number; if set to 0, pyramids are not used (single level), if set to 1, two levels are used, and so on                
                criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03), # parameter, specifying the termination criteria of the iterative search algorithm                
                flags = cv2.OPTFLOW_LK_GET_MIN_EIGENVALS,
                minEigThreshold = self.myShiTomasiNewFeaturesThreshold
         )# criteria -> after the specified maximum number of iterations criteria.maxCount or when the search window moves by less than criteria.epsilon

        p1Good = self.removeBadFeatures(p1, st, err)   

        return p1Good

    #---------------------------------------------------------------------------------------------------------------------

    def removeBadFeatures(self, p1, st, err):

        # print(f"{p1=}")
        # print(f"{st=}")	
        # print(f"{err=}")
    
        status0List = []
        
        for errIdx, errEle in enumerate(err):
            if errEle[0] > self.opticalFlowErrorThreshold or st[errIdx] == [0] or \
             (not self.y0 <= p1[errIdx][0][1] <= self.y1) or (not self.x0 <= p1[errIdx][0][0] <= self.x1):
                status0List.append(errIdx)

        p1Good = np.delete(p1,status0List, axis=0)
        
        return p1Good

    #---------------------------------------------------------------------------------------------------------------------

    def drawFeatures(self, frame, newFeatures, oldFeatures):
         
        for idx, (new, old) in enumerate(zip(newFeatures[0:self.maxNumOfFeaturesShown], oldFeatures[0:self.maxNumOfFeaturesShown])): 
				
            p1X, p1Y = new.ravel().astype(np.uint32)
            p0X, p0Y = old.ravel().astype(np.uint32)

            self.linesMask = cv2.line(self.linesMask, (p1X, p1Y), (p0X, p0Y), self.color[idx].tolist(), 2)

            frame = cv2.circle(frame, (p1X, p1Y), 5, self.color[idx].tolist(), -1)
    
        if self.to_draw_line:
            frame = cv2.add(frame, self.linesMask)
        return frame

    #---------------------------------------------------------------------------------------------------------------------

if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(Tracker)
    unittest.TextTestRunner(verbosity=0).run(suite)





                # print(f"{newFeatruePoints=}")
                # print(f"{p0=}")
                # if type(newFeatruePoints) != type(None) and newFeatruePoints.size > p0.size:
                #     print("newFeatruePoints.size > p0.size")
                #     p0 = self.getClosestFeatures(newFeatruePoints, p0)
                # else:
                #     print("p0 same size")
                    
                #     p0 = newFeatruePoints


    # #---------------------------------------------------------------------------------------------------------------------

    # def getClosestFeatures(self, newFeatruePoints, p0):

    #     p0Good = []
    #     for oldP0 in p0:

    #         minDistance = np.Inf
    #         for newIdx, newP0 in enumerate(newFeatruePoints):

    #             dist = np.linalg.norm(newP0-oldP0)
    #             if dist < minDistance:
    #                 minDistanceIdx = newIdx
    #         p0Good.append(newFeatruePoints[minDistanceIdx])
    #     p0Good = np.array(p0Good)

    #     return p0Good




    #---------------------------------------------------------------------------------------------------------------------

    # def searchLargerRoi(self, newFrame):
    #     newFeatures = None
    #     span = 1
    #     while type(newFeatures) == type(None):
    #         span *= 2
    #         roiMask = np.zeros((self.frameHeight, self.frameWidth), dtype=np.uint8)
    #         roiMask[self.y0 - span:self.y1 + span, self.x0 - span:self.x1 + span] = 255

    #         newFeatures = self.getImageFeatures(newFrame, roiMask)

    #     return newFeatures




    #---------------------------------------------------------------------------------------------------------------------
'''         cv2.goodFeaturesToTrack uses the min eigen value and takes qualityLevel*maxValue rather than above a threshold as implemented above

    def getImageFeatures(self, img, roiMask = None):
        # This function returns all features with threshold > (qualityLevel* max_value_feature), rather than absolute val as threshold
        #https://docs.opencv.org/3.4/db/d27/tutorial_py_table_of_contents_feature2d.html
        # goodFeaturesToTrack 2 options: Shi-Tomasi or Harris corner score
        # By default using Shi-Tomasi algorithm (useHarrisDetector = False), based on Harris corner detection.
        if type(roiMask) != type(None):
            roiMask = roiMask
        else:
            roiMask = self.roiMask

        imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        p0 = cv2.goodFeaturesToTrack(   # Outputs vector of detected corners.
            image = imgGray,            # Input 8-bit or floating-point 32-bit, single-channel image.              
            maxCorners = 100,           # Maximum number of corners to return (-1 for all). If there are more corners than are found, the strongest of them is returned.
            qualityLevel = 0.3,         # minimal accepted quality of image corners. qualityLevel is multiplied by max(entire image R): Shi-Tomasi or Harris, depending on useHarrisDetector
            minDistance = 7,           # Minimum possible Euclidean distance between the returned corners.
            mask = roiMask,                # Optional region of interest. It specifies the region in which the corners are detected.
            blockSize = 7,             # Size of an average block for computing a derivative covariation matrix over each pixel neighborhood.
            useHarrisDetector = False,  # False - use cornerMinEigenVal, True - use cornerHarris
            k = None,                   # 0.04 Free parameter of the Harris detector.relevant only when useHarrisDetector = True
            )

        return p0

        # cv2.cornerMinEigenVal and cv2.cornerHarris works similary:
        # For every pixel p in an image, the functions considers a blockSize × blockSize neighborhood S(p)
        # It calculates the covariation matrix of derivatives over the neighborhood as 2x2 matrix M, and from M calcultes a score R.
        # Shi-Tomasi score R=min(λ1,λ2) using cv2.cornerMinEigenVal and Harris function response score R=det(M)−k(trace(M))2 using cv2.cornerHarris
        # For each pixel in the image: cv2.cornerMinEigenVal retun R=the minimal eigen value and cv2.cornerHarris returns R=det(M)−k(trace(M))2

        # goodFeaturesToTrack takes cv2.cornerMinEigenVal and cv2.cornerHarris results image (image with pixels as the results R).
        # Performs a non-maximum suppression (the local maximums in 3 x 3 neighborhood are retained) over the image and finds the corner with the maximum value
        # Then, thresholds the corners (the R value) with (qualityLevel *  maximum value). corners less than this value are rejected.
        # The remaining corners are sorted by the quality measure in the descending order.
        # Function throws away each corner for which there is a stronger corner at a distance less than maxDistance.

'''