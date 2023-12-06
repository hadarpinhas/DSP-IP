from argparse import RawDescriptionHelpFormatter
from   pathlib              import Path
import unittest
import time
import cv2
import numpy as np
import sys
import math
import os
from matplotlib import pyplot as plt

'''
Object detection code, specificallty for IR images (grayscale), but not limited to.
The run time can be improved by updating the self.findBbox: 1. not to run on the entire image, 2. not for large different sizes of objects
'''

class ObjectDetection(unittest.TestCase):
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
               
        # basePath = r"/home/yossi/Documents/database/hadar" # for yossi pc
        basePath = r"C:\Users\User\Documents\dataBase\DSP-IP" # for Hadar's personal pc

        # relPath = r"videos/IR_Videos/output_drone1_1m.mp4"                # input path
        relPath = r"videos/IR_Videos/drone1_1m.mp4"                         # input path

        outRelPath = r"videos/outputVideos/output_drone1_1m.mp4"

        videoPath = os.path.join(basePath , relPath) 
        self.outputVideoPath = os.path.join(basePath , outRelPath)
        print(f"{videoPath=}")
        print(f"{self.outputVideoPath=}")

        self.fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.outputFps = 30.0

        self.cap = cv2.VideoCapture(videoPath)
        self.startDetectionFrame = 0 # 

        self.waitKeyParameter = 1 # in milliseconds, -1/0 for paused, alternatively, 1 milisecond continuous

        self.to_draw_line = False        

        self.screenSizeFactor = 1 # 0.9 for my laptop smaller screen

        self.frameCutFactor = 1 # to cut annotations built-in on images like the word "FLIR" and other trademarks...
        self.cutSize        = 100 # the number of pixels to cut to eliminate any annotations 

        self.imageDownScaleFactor = 0.05 # for detection in images with large pixels size (low resolution)

    #---------------------------------------------------------------------------------------------------------------------

    def runTracker(self):

        self.initVideoParams()

        while(True):
            _, newFrame = self.cap.read()

            if type(newFrame) == type(None):
                break
            if self.screenSizeFactor != 1: # for my laptop smaller screen
                newFrame = cv2.resize(newFrame, None, fx=self.screenSizeFactor,fy=self.screenSizeFactor)

            if self.frameCutFactor != 1:
                newFrame = newFrame[newFrame.shape[0] - self.cutSize : self.cutSize, newFrame.shape[1] - self.cutSize  : self.cutSize ]

            frameGray = cv2.cvtColor(newFrame, cv2.COLOR_BGR2GRAY)

            pMax = self.findBbox(frameGray)

            annotatedFrame = cv2.circle(newFrame, center=pMax, radius=4 ,color=(0,0,255), thickness=2)

            # self.createSmallerTarget(newFrame, pMax) # for creating a new video with a different synthetic drone.

            cv2.imshow('frame', cv2.resize(newFrame, None, fx=0.9,fy=0.9)) 

            # self.outputVideo.write(newFrame)
               
            k = cv2.waitKey(self.waitKeyParameter)
            if k == 27: 
                break

        cv2.destroyAllWindows() 
        self.cap.release()
        self.outputVideo.release()

    #---------------------------------------------------------------------------------------------------------------------

    def createSmallerTarget(self, newFrame, pMax):

        cv2.circle(newFrame, center=pMax, radius=20, color=(5,5,5), thickness=-1)
        dgl = 150 # drone Gray Level
        cv2.circle(newFrame, center=pMax, radius=max(1, 2*self.imageDownScaleFactor),color=(dgl,dgl,dgl), thickness=-1)

    #---------------------------------------------------------------------------------------------------------------------
       
    def initVideoParams(self):

        frameNumber = 0
        success, firstFrame = self.cap.read()
        while(frameNumber < self.startDetectionFrame) and success: # skip irrelevant frames
            _, firstFrame = self.cap.read()            
            frameNumber += 1

        if not success:
            sys.exit("No first Frame! chack data/video (data source path) please.")

        if self.frameCutFactor != 1:
            frameWidth, frameHeight = int(firstFrame.shape[1] * self.frameCutFactor), int(firstFrame.shape[0] * self.frameCutFactor)
            firstFrame = firstFrame[firstFrame.shape[0] - self.frameHeight : self.frameHeight, firstFrame.shape[1] - self.frameWidth  : self.frameWidth ]
        else:
            frameWidth, frameHeight = firstFrame.shape[1], firstFrame.shape[0]

        self.outputVideo = cv2.VideoWriter(filename=self.outputVideoPath, fourcc=self.fourcc, fps=self.outputFps, frameSize=(firstFrame.shape[1], firstFrame.shape[0]))

    #---------------------------------------------------------------------------------------------------------------------
        
    def findBbox(self, frameGray):

        kernelSize = 31
        max_val_max = 0
        boxSideList = [1, 3, 5, 7] # going over different object sizes to match template with
        for boxSide in boxSideList:
            kernel = np.zeros((kernelSize,kernelSize),np.uint8)
            cv2.circle(kernel, center=(kernelSize//2,kernelSize//2), radius=boxSide,color=(255,255,255), thickness=-1)

            corrMat = cv2.matchTemplate(image=frameGray, templ=kernel, method=cv2.TM_CCORR_NORMED)

            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(corrMat)

            if max_val > max_val_max: # tke the best correlation coeeficient (max_val) of all the boxSide
                max_val_max = max_val
                max_loc_max = max_loc

        pMax = max_loc_max[0] + kernelSize//2, max_loc_max[1] + kernelSize//2 # correlation matrrix (corrMat) is smaller than image so adding the difference

        pMax = self.getFineTune(frameGray, pMax) # get the center of objet/blob

        return pMax

    #---------------------------------------------------------------------------------------------------------------------
    
    def getFineTune(self, frameGray, pMax):
            
        fineTuneMatchBox = 100 # search center of blob in this area 100x100 around the object estimated location 

        frameGraySub = frameGray[ pMax[1]-fineTuneMatchBox//2:pMax[1]+fineTuneMatchBox//2,
                                  pMax[0]-fineTuneMatchBox//2:pMax[0]+fineTuneMatchBox//2 ]

        ret,thresh = cv2.threshold(frameGraySub,30,255,0)  

        M = cv2.moments(thresh)     
        # calculate x,y coordinate of center mji=∑x,y(array(x,y)⋅xj⋅yi), xj=x^j. yi=y^i
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])

        pMax = pMax[0] + cX - fineTuneMatchBox//2, pMax[1] + cY - fineTuneMatchBox//2 # like with the correlation matrrix, adding the difference

        return pMax

    #---------------------------------------------------------------------------------------------------------------------

if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(ObjectDetection)
    unittest.TextTestRunner(verbosity=0).run(suite)
