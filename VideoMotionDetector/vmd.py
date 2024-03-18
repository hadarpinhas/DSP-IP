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
# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", help="path to the video file", 
				# default=r'/home/yossi/Documents/database/hadar/videos/office_Vmd/officeVmd.mp4')
	default=r'/home/yossi/Documents/database/hadar/videos/smallObjTraffic/trafficVmd.mp4')
ap.add_argument("-min", "--min-area", type=int, default=10, help="minimum area size")
ap.add_argument("-max", "--max-area", type=int, default=20, help="minimum area size")
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

buffer = deque(maxlen=10) # prepera que for 10 last frames

# loop over the frames of the video
while True:
	# grab the current frame and initialize the occupied/unoccupied text
	frame = vs.read()
	frame = frame if args.get("video", None) is None else frame[1]

	# print(f"{frame.shape=}") # 1920 x 1080 x 3
	text = "Unoccupied"
	# if the frame could not be grabbed, then we have reached the end
	# of the video
	if frame is None:
		break
	# resize the frame, convert it to grayscale, and blur it
	# frame = imutils.resize(frame, width=500)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	gray = cv2.GaussianBlur(gray, (11, 11), 0)
	# if the first frame is None, initialize it
	if prevFrame is None:
		prevFrame = gray
		continue
	
    	# compute the absolute difference between the current frame and
	# prevFrame frame
	frameDelta = cv2.absdiff(prevFrame, gray)
	thresh = cv2.threshold(frameDelta, 20, 255, cv2.THRESH_BINARY)[1]
	
	# dilate the thresholded image to fill in holes, then find contours on thresholded image
	# thresh = cv2.dilate(thresh, None, iterations=3)
	thresh = cv2.morphologyEx(src=thresh, op=cv2.MORPH_OPEN, kernel=(5,5))
	cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
	cnts = imutils.grab_contours(cnts)
	
	# loop over the contours
	for c in cnts:
		# if the contour is too small, ignore it
		if cv2.contourArea(c) < args["min_area"]:
		# if cv2.contourArea(c) > args["max_area"]:
			continue
		# compute the bounding box for the contour, draw it on the frame, and update the text
		(x, y, w, h) = cv2.boundingRect(c)
		cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)
		text = "Occupied"
		
    # draw the text and timestamp on the frame
	# cv2.putText(frame, "Room Status: {}".format(text), (10, 20),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
	cv2.putText(frame, datetime.datetime.now().strftime("%A %d %B %Y %I:%M:%S%p"),(10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
	# show the frame and record if the user presses a key
	cv2.imshow("Security Feed", frame)
	cv2.imshow("Thresh", thresh)
	cv2.imshow("Frame Delta", frameDelta*10)
	cv2.imshow("prevFrame", prevFrame)

	# if the `q` key is pressed, break from the lop
	if cv2.waitKey(0) & 0xFF == ord("q"):
		break
	outputVideo_frame.write(frame)
	# outputVideo_thresh.write(thresh)
	# outputVideo_frameDelta.write(frameDelta)

	buffer.append(gray)
	meanFrame = np.zeros(gray.shape)
	for f in buffer:
		cv2.accumulate(f,meanFrame)
	# print(f"{meanFrame.shape=}")
	# print(f"{len(buffer)=}")
	prevFrame = (meanFrame / len(buffer)+1).astype(np.uint8)

	# print(f"{prevFrame.shape=}")

	
# cleanup the camera and close any open windows
vs.stop() if args.get("video", None) is None else vs.release()

cv2.destroyAllWindows()

outputVideo_frame.release()		
# outputVideo_thresh.release()		
# outputVideo_frameDelta.release()		

print(f"video saved at: {os.path(__file__)=}")
