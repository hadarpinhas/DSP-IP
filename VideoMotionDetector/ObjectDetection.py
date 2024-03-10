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
import datetime

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

        basePath = Path(r"/home/yossi")
        # basePath = r"C:\Users\User\"

        relPath = Path('Documents/database/hadar/videos/teaser/wiggle_0.mp4')

        pathName = relPath.stem# A/B/name.png -> name
        pathParent = relPath.parents[0] # A/B/name.png -> A/B

        curTime = datetime.datetime.now().strftime("%d%m%Y_%H%M%S")
        outRelPath = str(pathParent) + '\\' + str(pathName) + '_' + curTime + ".mp4"

        videoPath = basePath / relPath 
        self.outputVideoPath = os.path.join(basePath , outRelPath)
        print(f"{videoPath=}")
        print(f"{self.outputVideoPath=}")

        self.fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.outputFps = 30.0

        self.cap = cv2.VideoCapture(str(videoPath))
        self.startDetectionFrame = 600 # 

        self.waitKeyParameter = 100 # in milliseconds, -1/0 for paused, alternatively, 1 milisecond continuous

        self.to_draw_line = False        

        self.screenSizeFactor = 1 # 0.9 for my laptop smaller screen

         # to cut annotations built-in on images like the word "FLIR" and other trademarks...
        self.cutSize        = 100 # the number of pixels to cut to eliminate any annotations 

        self.use_custom_kernels = False # whether to to use white (or black) box (or circle). False: crop template
        self.to_fine_tune   = True

        self.imageDownScaleFactor = 0.05 # for detection in images with large pixels size (low resolution)

    #---------------------------------------------------------------------------------------------------------------------

    def runTracker(self):

        self.initVideoParams()

        success, newFrame = self.cap.read()
        assert success==True

        templates = [np.nan if self.use_custom_kernels else self.findTemplate(cv2.cvtColor(newFrame, cv2.COLOR_BGR2GRAY))]      

        while(True):
            success, newFrame = self.cap.read()
            assert success==True
      
            if self.screenSizeFactor != 1: # for my laptop smaller screen
                newFrame = cv2.resize(newFrame, None, fx=self.screenSizeFactor,fy=self.screenSizeFactor)

            if newFrame.ndim == 3:
                frameGray = cv2.cvtColor(newFrame, cv2.COLOR_BGR2GRAY)

            if self.cutSize > 0:
                cleanFrame = frameGray[self.cutSize:frameGray.shape[0]-self.cutSize, self.cutSize:frameGray.shape[1]-self.cutSize]

            pMax = self.findBbox(cleanFrame, templates)

            pMax = pMax[0] + self.cutSize, pMax[1] + self.cutSize # x,y

            cv2.circle(newFrame, center=pMax, radius=10 ,color=(0,0,255), thickness=3)

            # self.createSmallerTarget(newFrame, pMax) # for creating a new video with a different synthetic drone.

            cv2.imshow('frame',newFrame) # cv2.resize(newFrame, None, fx=0.9,fy=0.9)) 

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

        if self.screenSizeFactor != 1:
            frameWidth, frameHeight = int(firstFrame.shape[1]*self.screenSizeFactor), int(firstFrame.shape[0]*self.screenSizeFactor)
            firstFrame = firstFrame[firstFrame.shape[0]-frameHeight: frameHeight, firstFrame.shape[1]-frameWidth: frameWidth ]
        else:
            frameWidth, frameHeight = firstFrame.shape[1], firstFrame.shape[0]

        self.outputVideo = cv2.VideoWriter(filename=self.outputVideoPath, fourcc=self.fourcc, fps=self.outputFps, frameSize=(firstFrame.shape[1], firstFrame.shape[0]))

    #---------------------------------------------------------------------------------------------------------------------
      
    def getKernels(self, kernelSize:int=111, polarity:int=-1, templatesSize:list=[41], objectShape:str='circles') -> list:

        kernels = []
        kernelSize = 111 # 31
        templatesSize = [41] # [1, 3, 5, 7] # going over different object sizes to match template with
        for boxSide in templatesSize:

            polarity    = -1 # 1 for bright objects and -1 for dark objects
            if polarity == 1:
                kernel      =  np.zeros((kernelSize,kernelSize),np.uint8)
                objectColor = (255,255,255)
            else: # polarity = -1
                kernel      = 255 * np.ones((kernelSize,kernelSize),np.uint8)
                objectColor = (0,0,0)

            objectShape = 'circle'
            if objectShape == 'circle':
                cv2.circle(kernel, center=(kernelSize//2,kernelSize//2), radius=boxSide, color=objectColor, thickness=-1)
            elif objectShape == 'rect':
                cv2.rectangle(kernel, pt1=(kernelSize//4,kernelSize//4), pt2=(kernelSize*3//4,kernelSize*3//4), color=objectColor, thickness=-1)

            kernels.append(kernel)

        return kernels
    
    #---------------------------------------------------------------------------------------------------------------------
       
    def findTemplate(self, frameGray:np.ndarray) -> np.ndarray:

        xi, yi, bboxWidth, bboxHeight = cv2.selectROI(frameGray)

        x0, x1 = xi, xi + bboxWidth
        y0, y1 = yi, yi + bboxHeight

        template = frameGray[y0:y1, x0:x1]

        ret, template = cv2.threshold(template, 100, 255,type=cv2.THRESH_TOZERO)

        templates = []
        for upscaleFactor in np.arange(1,5,0.5):
            templates.append(cv2.resize(template, None, fx=upscaleFactor, fy=upscaleFactor, interpolation=cv2.INTER_CUBIC)) 
            cv2.imshow('template', templates[-1])
            cv2.waitKey(0)

        return template

    #---------------------------------------------------------------------------------------------------------------------
        
    def findBbox(self, frameGray:np.ndarray, templates: list[np.ndarray] = [np.nan]) -> tuple:
        """uses the match template or generated templates of white (or black) box (or circle) in contrasted kernel
        """
        if self.use_custom_kernels:
            templates = self.getKernels()

        max_val_max = 0

        # frameGray = cv2.GaussianBlur(frameGray, (9,9), sigmaX=5)

        for template in templates:

            corrMat = cv2.matchTemplate(image=frameGray, templ=template, method=cv2.TM_CCORR_NORMED)

            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(corrMat) # return mxLoc as (x0, y0)

            # cv2.circle(frameGray, center=max_loc, radius=10, color=(255,255,255), thickness=-1)
            # cv2.imshow('frameGray', frameGray)
            # cv2.waitKey(0)

            if max_val > max_val_max: # take the best correlation coefficient (max_val) of all the boxSide
                max_val_max = max_val
                max_loc_max = max_loc
                shapeX, shapeY = template.shape[1], template.shape[0]

        # correlation matrrix (corrMat) is smaller than image so adding the difference
        pMax = (max_loc_max[0] + shapeX//2, max_loc_max[1] + shapeY//2)

        if self.to_fine_tune:
            pMax = self.getFineTune(frameGray, pMax) # get the center of object/blob

        return pMax

    #---------------------------------------------------------------------------------------------------------------------

    @staticmethod
    def getFineTune(frameGray: np.ndarray, pMax: tuple) -> tuple:
        """get the center of mass of all the correlation coefficient peaks"""

        fineTuneMatchBox = 100 # search center of blob in this area 100x100 around the object estimated location 

        frameGraySub = frameGray[ pMax[1]-fineTuneMatchBox//2:pMax[1]+fineTuneMatchBox//2,
                                  pMax[0]-fineTuneMatchBox//2:pMax[0]+fineTuneMatchBox//2 ]

        ret,thresh = cv2.threshold(frameGraySub,30,255,0)  

        M = cv2.moments(thresh)  
        if M["m00"] > 0:    
            # calculate x,y coordinate of center mji=∑x,y(array(x,y)⋅xj⋅yi), xj=x^j. yi=y^i
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])

            pMax = (pMax[0] + cX - fineTuneMatchBox//2, pMax[1] + cY - fineTuneMatchBox//2) # like with the correlation matrrix, adding the difference

        return pMax

    #---------------------------------------------------------------------------------------------------------------------

if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(ObjectDetection)
    unittest.TextTestRunner(verbosity=0).run(suite)
