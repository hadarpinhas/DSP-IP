# import the necessary packages
from imutils.video import VideoStream
from imutils.video import FPS
import argparse
import imutils
import time
import cv2

# from mtcnn_opencv.mtcnn_cv2 import MTCNN
from yoloFaceRepo import yoloface

import numpy as np
from PIL import Image

# model = YoloDetector(target_size=720, device="cuda:0", min_face=90)
# orgimg = np.array(Image.open('test_image.jpg'))
# bboxes,points = model.predict(orgimg)

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", type=str, default="faceVidCut.mp4",
    help="path to input video file")
ap.add_argument("-t", "--tracker", type=str, default='boosting', #"kcf",
    help="OpenCV object tracker type")
args = vars(ap.parse_args())


# extract the OpenCV version info
(major, minor) = cv2.__version__.split(".")[:2]
# if we are using OpenCV 3.2 OR BEFORE, we can use a special factory
# function to create our object tracker
if int(major) == 3 and int(minor) < 3:
    tracker = cv2.Tracker_create(args["tracker"].upper())
# otherwise, for OpenCV 3.3 OR NEWER, we need to explicity call the
# approrpiate object tracker constructor:
else:
    # initialize a dictionary that maps strings to their corresponding
    # OpenCV object tracker implementations
    OPENCV_OBJECT_TRACKERS = {
        "csrt": cv2.TrackerCSRT_create,
        "kcf": cv2.TrackerKCF_create,
        "boosting": cv2.legacy.TrackerBoosting_create,
        "mil": cv2.TrackerMIL_create,
        "tld": cv2.legacy.TrackerTLD_create,
        "medianflow": cv2.legacy.TrackerMedianFlow_create,
        "mosse": cv2.legacy.TrackerMOSSE_create
    }
    # grab the appropriate object tracker using our dictionary of
    # OpenCV object tracker objects
    # tracker = OPENCV_OBJECT_TRACKERS[args["tracker"]]()

# if a video path was not supplied, grab the reference to the web cam
if not args.get("video", False):
    print("[INFO] starting video stream...")
    vs = VideoStream(src=0).start()
    time.sleep(1.0)
# otherwise, grab a reference to the video file
else:
    print(f"get the video: {args['video']}")
    vs = cv2.VideoCapture(args["video"])
# initialize the FPS throughput estimator
fps = None

trackers = {} # lsit of trackers 

success, firstFrame = vs.read()

if success:
    # initBB = cv2.selectROI(firstFrame)

    image = cv2.cvtColor(firstFrame, cv2.COLOR_BGR2RGB)
    image = imutils.resize(image, width=500)
    result = yoloface._getBBox(image)
    print(f"{result=}")
    
    for idx, obj in enumerate(result):
        tracker = OPENCV_OBJECT_TRACKERS[args["tracker"]]()
        initBB = result[idx] # initialize the bounding box coordinates of the object we are going to track   
        tracker.init(firstFrame, initBB)
        trackers[idx] = tracker
print(f"{trackers=}")

# loop over frames from the video stream
while True:
    # grab the current frame, then handle if we are using a
    # VideoStream or VideoCapture object
    frame = vs.read() # vs.read yields a tuple (success, frame), that is why next line take [1]
    frame = frame[1] if args.get("video", False) else frame
    # check to see if we have reached the end of the stream
    if frame is None:
        break
    # resize the frame (so we can process it faster) and grab the
    # frame dimensions
    frame = imutils.resize(frame, width=500)
    (H, W) = frame.shape[:2]
    

        # check to see if we are currently tracking an object
    if initBB is not None:
        for tracker in trackers.values():
            # print(f"{tracker=}")
        # grab the new bounding box coordinates of the object
            (success, box) = tracker.update(frame)
            # check to see if the tracking was a success
            if success:
                (x, y, w, h) = [int(v) for v in box]
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # update the FPS counter
        if not fps:
            fps = FPS().start()
        fps.update()
        fps.stop()
        # initialize the set of information we'll be displaying on
        # the frame
        info = [
            ("Tracker", args["tracker"]),
            ("Success", "Yes" if success else "No"),
            ("FPS", "{:.2f}".format(fps.fps())),
        ]
        # loop over the info tuples and draw them on our frame
        for (i, (k, v)) in enumerate(info):
            text = "{}: {}".format(k, v)
            cv2.putText(frame, text, (10, H - ((i * 20) + 20)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

                # show the output frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(50) & 0xFF
    # if the 's' key is selected, we are going to "select" a bounding
    # box to track
    if key == ord("s"):
        # select the bounding box of the object we want to track (make
        # sure you press ENTER or SPACE after selecting the ROI)
        initBB = cv2.selectROI("Frame", frame, fromCenter=False,
            showCrosshair=True)
        # start OpenCV object tracker using the supplied bounding box
        # coordinates, then start the FPS throughput estimator as well
        tracker.init(frame, initBB)
        fps = FPS().start()
        
            # if the `q` key was pressed, break from the loop
    elif key == ord("q"):
        break
# if we are using a webcam, release the pointer
if not args.get("video", False):
    vs.stop()
# otherwise, release the file pointer
else:
    vs.release()
# close all windows
cv2.destroyAllWindows()