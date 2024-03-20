# https://pyimagesearch.com/2015/05/25/basic-motion-detection-and-tracking-with-python-and-opencv/#pyis-cta-modal

# import the necessary packages
from imutils.video import VideoStream
import argparse
import datetime
import imutils
import time
import cv2
import os
import numpy as np
from collections import deque
import utils
# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", help="path to the video file", 
				# default=r'/home/yossi/Documents/database/hadar/videos/office_Vmd/officeVmd.mp4')
	# default=r'/home/yossi/Documents/database/hadar/videos/smallObjTraffic/trafficVmd.mp4')
	# default=r'C:\Users\User\Documents\dataBase\DSP-IP\videos\smallObjTraffic/trafficVmd.mp4')
	default=r'/home/yossi/Documents/database/hadar/videos/smallObjTraffic/trafficVidCamShift.mp4')

ap.add_argument("-min", "--min-area", type=int, default=100, help="minimum area size")
ap.add_argument("-max", "--max-area", type=int, default=1000, help="minimum area size")
args = vars(ap.parse_args())
# if the video argument is None, then we are reading from webcam
if args.get("video", None) is None:
	vs = VideoStream(src=0).start()
	time.sleep(2.0)
# otherwise, we are reading from a video file
else:
	vs = cv2.VideoCapture(args["video"])
# initialize the first frame in the video stream
prevFrame = None

basePath = args['video'][:-4]

curTime = datetime.datetime.now().strftime("%d%m%Y_%H%M%S")
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
outputName_frame        = basePath + curTime + '_frame' + '.mp4'
# outputName_thresha      = basePath + curTime + '_thresh' + '.mp4'
# outputName_frameDelta   = basePath + curTime + '_frameDelta' + '.mp4'
w,h =  1920, 1080 #500, int(1920/1080 * 500)
outputVideo_frame       = cv2.VideoWriter(filename=outputName_frame,       fourcc=fourcc, fps=30, frameSize=(w, h))
# outputVideo_thresh      = cv2.VideoWriter(filename=outputName_thresha,     fourcc=fourcc, fps=30, frameSize=(w, h))
# outputVideo_frameDelta  = cv2.VideoWriter(filename=outputName_frameDelta,  fourcc=fourcc, fps=30, frameSize=(w, h))

buffer = deque(maxlen=5) # prepera que for x last frames

# loop over the frames of the video
freeRun = 100
for i in range(freeRun):
	success, frame = vs.read()
assert success, "no video"
centerX, centerY = frame.shape[1]//2, frame.shape[0]//2

def normalizeImage(img: np.ndarray) -> np.ndarray:
    return ((img - img.min()) / (img.max() - img.min())) * 255

def matchTemplateBackShift(templateImg, gray):
	max_val, max_loc = utils.match(baseImg=gray, templateImg=f)

	translation_matrix = np.float32([ [1,0,max_loc[0]], [0,1,max_loc[1]] ])

	img_translated = cv2.warpAffine(f, translation_matrix, gray.shape[1::-1])

	return img_translated

def opticalFlowBackShift(templateImg, gray):
	flow = utils.opticalFlow(prev=templateImg , next=gray)

	translation_matrix = np.float32([ [1,0,flow[1]], [0,1,flow[0]] ])

	img_translated = cv2.warpAffine(f, translation_matrix, gray.shape[1::-1])

	return img_translated

while True:
	# grab the current frame and initialize the occupied/unoccupied text
	frame = vs.read()
	frame = frame if args.get("video", None) is None else frame[1] # consider device live camera

	if frame is None:
		break

	frame = cv2.getRectSubPix(image=frame, patchSize=(500, 500), center=(centerX, centerY))

	# if the frame could not be grabbed, then we have reached the end of the video

	# resize the frame, convert it to grayscale, and blur it
	# frame = imutils.resize(frame, width=500)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	gray = cv2.GaussianBlur(gray, (11, 11), 0)

	meanFrameDelta = np.zeros(gray.shape, dtype=np.float32)
	for i,f in enumerate(buffer):

		# img_translated = matchTemplateBackShift(f, gray)

		img_translated = opticalFlowBackShift(f, gray)

		frameDelta = cv2.absdiff(img_translated, gray)

		meanFrameDelta += frameDelta

	buffer.append(gray)

	meanFrameDelta = meanFrameDelta[:,30:-30]
	meanFrameDelta = (meanFrameDelta//(len(buffer)))
	if meanFrameDelta.any():
		print("normalizeImage(meanFrameDelta)")
		print(f"{np.max(meanFrameDelta)=}")
		meanFrameDelta = normalizeImage(meanFrameDelta)
		print(f"{np.max(meanFrameDelta)=}")
	meanFrameDelta = meanFrameDelta.astype(np.uint8)

	thresh = cv2.threshold(meanFrameDelta, 200, 255, cv2.THRESH_BINARY)[1]

	
	# dilate the thresholded image to fill in holes, then find contours on thresholded image
	thresh = cv2.dilate(thresh, None, iterations=10)
	# thresh = cv2.morphologyEx(src=thresh, op=cv2.MORPH_CLOSE, kernel=(5,5), iterations=3)
	cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
	cnts = imutils.grab_contours(cnts)
	print(f"{len(cnts)=}")
	
	# loop over the contours
	for c in cnts:
		# if the contour is too small, ignore it
		# if :
		print(f"{cv2.contourArea(c)=}")
		if cv2.contourArea(c) < args["min_area"] or cv2.contourArea(c) > args["max_area"]:
			continue
		# compute the bounding box for the contour, draw it on the frame, and update the text
		(x, y, w, h) = cv2.boundingRect(c)
		cv2.rectangle(frame, (x-max_loc[0]*len(buffer), y-max_loc[1]*len(buffer)), (x-max_loc[0]*len(buffer) + w, y-max_loc[1]*len(buffer) + h), (0, 255, 0), 1)

		
    # draw the text and timestamp on the frame
	# cv2.putText(frame, "Room Status: {}".format(text), (10, 20),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
	cv2.putText(frame, datetime.datetime.now().strftime("%A %d %B %Y %I:%M:%S%p"),(10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
	# show the frame and record if the user presses a key
	cv2.imshow("frame", frame)
	cv2.imshow("meanFrameDelta", meanFrameDelta)
	cv2.imshow("Thresh", thresh)

	# if the `q` key is pressed, break from the lop
	if cv2.waitKey(0) & 0xFF == ord("q"):
		break
	# outputVideo_frame.write(frame)

# cleanup the camera and close any open windows
vs.stop() if args.get("video", None) is None else vs.release()

cv2.destroyAllWindows()

outputVideo_frame.release()		
# outputVideo_thresh.release()		
# outputVideo_frameDelta.release()		

print(f"video saved at: {os.path.abspath(__file__)=}")
