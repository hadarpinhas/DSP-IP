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

        relPath = Path('Documents/database/hadar/videos/teaser/wiggleIn.mp4')   

        pathName = relPath.stem# A/B/name.png -> name
        pathParent = relPath.parents[0] # A/B/name.png -> A/B

        curTime = datetime.datetime.now().strftime("%d%m%Y_%H%M%S")
        outRelPath = pathParent / (pathName + '_' + curTime + ".mp4")

        videoPath = self.basePath / relPath 
        self.outputVideoPath = self.basePath / outRelPath
        print(f"{videoPath=}")
        print(f"{self.outputVideoPath=}")

        self.fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.outputFps = 30.0

        self.cap = cv2.VideoCapture(str(videoPath))
        self.startDetectionFrame = 0 # 

        self.waitKeyParameter = 1 # in milliseconds, -1/0 for continuous

        self.to_draw_line = False        

        self.screenSizeFactor = 1 # 0.9 for my laptop smaller screen

    #---------------------------------------------------------------------------------------------------------------------

    def runTracker(self):

        self.initTracker()

        self.outputVideo = cv2.VideoWriter(filename=str(self.outputVideoPath), fourcc=self.fourcc, fps=self.outputFps, frameSize=(self.frameWidth,self.frameHeight))
	
        self.startTrakcerLoop()

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

        self.linesMask  = np.zeros_like((firstFrame))

        cv2.imshow('frame', firstFrame) 
        
    #---------------------------------------------------------------------------------------------------------------------

    def startTrakcerLoop(self):
        
        counter = 0
        while(True):
            success, frame = self.cap.read()
            if not success:
                break
            counter += 1

            if self.screenSizeFactor != 1: # for my laptop smaller screen
                frame = cv2.resize(frame, None, fx=self.screenSizeFactor,fy=self.screenSizeFactor)

            # Update tracker
            ok, bbox = self.tracker.update(frame)

            if ok:# Tracking success
                p1 = (int(bbox[0]), int(bbox[1]))
                p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))       


            cv2.rectangle(frame, p1, p2, (0, 255, 0), 1)
            cv2.imshow('frame', frame)
            if cv2.waitKey(self.waitKeyParameter) & 0xff == ord('q'): 
                break

            self.outputVideo.write(frame)         


        cv2.destroyAllWindows() 
        self.cap.release()
        self.outputVideo.release()

    #---------------------------------------------------------------------------------------------------------------------

if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(Tracker)
    unittest.TextTestRunner(verbosity=0).run(suite)

