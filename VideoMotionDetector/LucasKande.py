
# based on https://www.geeksforgeeks.org/python-opencv-optical-flow-with-lucas-kanade-method/
# cv2.goodFeaturesToTrack is based on Harris corner detection
# calcOpticalFlowPyrLK is absed on Lucas kenade
from pathlib import Path
from re import A 
import numpy as np 
import cv2 
import sys

# videoPath = Path(r"C:\Users\User\Documents\dataBase\DSP-IP\videos\videos\\1.h264")
videoPath = Path(r"C:\Users\User\Documents\dataBase\DSP-IP\videos\24_cut.ts")

# when working on my laptop I decreased the videos frames by x0.9 to fit my laptop screen
screenFactor = 1 # 0.9 for my laptop smaller screen

cap = cv2.VideoCapture(str(videoPath)) 

# params for corner detection 
feature_params = dict( maxCorners = 100, qualityLevel = 0.3, minDistance = 11, blockSize = 11) 

# Parameters for lucas kanade optical flow 
 # https://docs.opencv.org/3.4/dc/d6b/group__video__track.html#ga473e4b886d0bcc6b65831eb88ed93323
lk_params = dict( winSize = (3, 3), maxLevel = 2, criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)) 

# Create some random colors 
color = np.random.randint(0, 255, (100, 3)) 

# Take first frame and find corners in it 
ret, firstFrame = cap.read()
print(f"{firstFrame.shape=}")
if screenFactor < 1: # for my laptop smaller screen
	firstFrame = cv2.resize(firstFrame, None, fx=screenFactor,fy=screenFactor)	


zeroedImage = np.zeros_like(firstFrame)  # mask to add line onto

oldGray = cv2.cvtColor(firstFrame, cv2.COLOR_BGR2GRAY) 

xi, yi, bboxWidth, bboxHeight = cv2.selectROI(oldGray) # tuple (Top_Left_X, Top_Left_Y, Width, Height)
print(f"{xi=}, {yi=}, {bboxWidth=}, {bboxHeight=}")

x0, x1 = xi, xi + bboxWidth
y0, y1 = yi, yi + bboxHeight

zeroedImageMasked = np.zeros_like(oldGray)

zeroedImageMasked[y0:y1, x0:x1] = 255 # *np.ones((abs(y0-y1), abs(x0-x1)), dtype=np.float32)

p0 = cv2.goodFeaturesToTrack(oldGray, mask = zeroedImageMasked, **feature_params)
print(f"{p0=}")

while(True):

	ret, newFrame = cap.read()
	if screenFactor < 1:	# for my laptop smaller screen	
		newFrame = cv2.resize(newFrame, None, fx=screenFactor,fy=screenFactor)	

	newGray = cv2.cvtColor(newFrame, cv2.COLOR_BGR2GRAY)

	# calculate optical flow 
 	# prevImg, nextImg, prevPts, nextPts[, status[, err[, winSize[, maxLevel[, criteria[, flags[, minEigThreshold
	p1, st, err = cv2.calcOpticalFlowPyrLK(prevImg=oldGray, nextImg=newGray, prevPts=p0, nextPts=None, 
					 **lk_params) # p1=nextPoint (found-> feature p1=pos(x,y) not found p1=None), status(found feaure-> st[i]=[1], not foud st[i]=[0]), error
	print(f"{p1=}")
	print(f"{st=}")	
	print(f"{err=}")
		
	# if type(p1) != type(None):
		# Select good points 
	is_feature_detected = False	
	# print(f"{st=}")
	for errIx, errEle in enumerate(err):
		if 	errEle[0] > 10 or st[errIx] == [0]:
			print(f"{p1=}")			
			p1 = np.delete(p1,errIx, axis=0)
			print(f"deleting {errEle}")		
		
	print(f"{p1=}")
	print(f"{st=}")
	
	if len(p1) > 0:
		is_feature_detected = True	
		good_new = p1
		good_old = p0	
	else:
		good_new = p0 
		good_old = p0							
	print(f"{is_feature_detected=}")



	# # draw the tracks 
	for i, (new, old) in enumerate(zip(good_new, good_old)): 
				
		p1X, p1Y = new.ravel().astype(np.uint32)			
		p0X, p0Y = old.ravel().astype(np.uint32)

		zeroedImage = cv2.line(zeroedImage, (p1X, p1Y), (p0X, p0Y), color[i].tolist(), 2)
		
		newFrame = cv2.circle(newFrame, (p1X, p1Y), 5, color[i].tolist(), -1)
		
	img = cv2.add(newFrame, zeroedImage)
	# else:
	# 	img = newFrame		

	# # Updating Previous frame and points 
	oldGray = newGray.copy()
	zeroedImageMasked = np.zeros_like(oldGray)
	
	if not is_feature_detected:
		x0, x1 = p1X - bboxWidth //2, p1X + bboxWidth //2
		y0, y1 = p1Y - bboxHeight//2, p1Y + bboxHeight//2
			
		print(f"{x0=}")		
		print(f"{x1=}")		
		print(f"{y0=}")		
		print(f"{y1=}")		
		zeroedImageMasked[y0:y1, x0:x1] = 255 # *np.ones((abs(y0-y1), abs(x0-x1)), dtype=np.float32)
		
		p0 = cv2.goodFeaturesToTrack(oldGray, mask = zeroedImageMasked, **feature_params)
		
	elif type(p1) != type(None): # if p1 is None, no optical flow was found. if found then p1 becomes the old point.
		p0=p1
		
	print(f"{p0=}")		

	if x0 < 0 or y0 < 0 or x1 > oldGray.shape[1] or y1 > oldGray.shape[0]:
		sys.exit("out of screen!")

		
	cv2.imshow('frame', img) 
	
	k = cv2.waitKey(-1) 
	if k == 27: 
		break		

cv2.destroyAllWindows() 
cap.release() 

