
# based on https://www.geeksforgeeks.org/python-opencv-optical-flow-with-lucas-kanade-method/
# cv2.goodFeaturesToTrack is based on Harris corner detection
# calcOpticalFlowPyrLK is absed on Lucas kenade

import numpy as np 
import cv2 

cap = cv2.VideoCapture('1.h264') 

# params for corner detection 
feature_params = dict( maxCorners = 100, qualityLevel = 0.3, minDistance = 11, blockSize = 11) 

# Parameters for lucas kanade optical flow 
lk_params = dict( winSize = (7, 7), maxLevel = 2, criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)) 

# Create some random colors 
color = np.random.randint(0, 255, (100, 3)) 

# Take first frame and find corners in it 
ret, firstFrame = cap.read()

mask = np.zeros_like(firstFrame)  # mask to add line onto

oldGray = cv2.cvtColor(firstFrame, cv2.COLOR_BGR2GRAY) 
p0 = cv2.goodFeaturesToTrack(oldGray, mask = None, **feature_params) 

while(True):

	ret, newFrame = cap.read() 		
	newGray = cv2.cvtColor(newFrame, cv2.COLOR_BGR2GRAY) 

	# calculate optical flow 
	p1, st, err = cv2.calcOpticalFlowPyrLK(oldGray, newGray, p0, None, **lk_params) 

	if type(p1) != type(None):
		# Select good points 
		good_new = p1[st == 1] 
		good_old = p0[st == 1] 

		# # draw the tracks 
		for i, (new, old) in enumerate(zip(good_new, good_old)): 
				
			a, b = new.ravel().astype(np.uint32)			
			c, d = old.ravel().astype(np.uint32)

			mask = cv2.line(mask, (a, b), (c, d), color[i].tolist(), 2)
		
			newFrame = cv2.circle(newFrame, (a, b), 5, color[i].tolist(), -1) 
		
		img = cv2.add(newFrame, mask)
	else:
		img = newFrame		

	# # Updating Previous frame and points 
	oldGray = newGray.copy() 
	p0 = cv2.goodFeaturesToTrack(oldGray, mask = None, **feature_params) 
		
	cv2.imshow('frame', img) 
	
	k = cv2.waitKey(1) 
	if k == 27: 
		break		

cv2.destroyAllWindows() 
cap.release() 

