
# based on https://www.geeksforgeeks.org/python-opencv-optical-flow-with-lucas-kanade-method/
# cv2.goodFeaturesToTrack is based on Harris corner detection
# calcOpticalFlowPyrLK is absed on Lucas kenade
from pathlib import Path 
import numpy as np 
import cv2 

# videoPath = Path(r"C:\Users\User\Documents\dataBase\DSP-IP\videos\videos\\1.h264")
videoPath = Path(r"C:\Users\User\Documents\dataBase\DSP-IP\videos\24_cut.ts")

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
firstFrame = cv2.resize(firstFrame, None, fx=0.9,fy=0.9)	


zeroedImage = np.zeros_like(firstFrame)  # mask to add line onto

oldGray = cv2.cvtColor(firstFrame, cv2.COLOR_BGR2GRAY) 

bbox = cv2.selectROI(oldGray) # tuple (Top_Left_X, Top_Left_Y, Width, Height)
print(f"{bbox}")
bboxWidth, bboxHeight = bbox[2], bbox[3]

x0, x1 = bbox[0], bbox[0] + bboxWidth
y0, y1 = bbox[1], bbox[1] + bboxHeight

zeroedImageMasked = np.zeros_like(oldGray)

zeroedImageMasked[y0:y1, x0:x1] = 255 # *np.ones((abs(y0-y1), abs(x0-x1)), dtype=np.float32)

p0 = cv2.goodFeaturesToTrack(oldGray, mask = zeroedImageMasked, **feature_params)

while(True):

	ret, newFrame = cap.read()
	newFrame = cv2.resize(newFrame, None, fx=0.9,fy=0.9)	

	newGray = cv2.cvtColor(newFrame, cv2.COLOR_BGR2GRAY)

	# calculate optical flow 
 	# prevImg, nextImg, prevPts, nextPts[, status[, err[, winSize[, maxLevel[, criteria[, flags[, minEigThreshold
	p1, st, err = cv2.calcOpticalFlowPyrLK(prevImg=oldGray, nextImg=newGray, prevPts=p0, nextPts=None, 
					 **lk_params)
	print(f"{p1=}")
	print(f"{st=}")	
	print(f"{err=}")
		
	# if type(p1) != type(None):
		# Select good points 
	is_feature_detected = False	
	# print(f"{st=}")
	if st.any(0):
		is_feature_detected = True	

	if is_feature_detected:
		good_new = p1[st == 1] 
		good_old = p0[st == 1]
		

	# # draw the tracks 
	for i, (new, old) in enumerate(zip(good_new, good_old)): 
				
		a, b = new.ravel().astype(np.uint32)			
		c, d = old.ravel().astype(np.uint32)

		zeroedImage = cv2.line(zeroedImage, (a, b), (c, d), color[i].tolist(), 2)
		
		newFrame = cv2.circle(newFrame, (a, b), 5, color[i].tolist(), -1)
			
		
	img = cv2.add(newFrame, zeroedImage)
	# else:
	# 	img = newFrame		

	# # Updating Previous frame and points 
	oldGray = newGray.copy()
	zeroedImageMasked = np.zeros_like(oldGray)
	
	if not is_feature_detected:
		x0, x1 = a - bboxWidth //2, a + bboxWidth //2
		y0, y1 = c - bboxHeight//2, c + bboxHeight//2
			
		print(f"{x0=}")		
		print(f"{x1=}")		
		print(f"{y0=}")		
		print(f"{y1=}")		
		zeroedImageMasked[y0:y1, x0:x1] = 255 # *np.ones((abs(y0-y1), abs(x0-x1)), dtype=np.float32)
		
		p0 = cv2.goodFeaturesToTrack(oldGray, mask = zeroedImageMasked, **feature_params)
	elif type(p1) != type(None):
	 	p0=p1
		
	print(f"{is_feature_detected=}")		
	print(f"{p0=}")		

		
	cv2.imshow('frame', img) 
	
	k = cv2.waitKey(-1) 
	if k == 27: 
		break		

cv2.destroyAllWindows() 
cap.release() 

