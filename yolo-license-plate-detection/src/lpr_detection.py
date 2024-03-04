import cv2 as cv
import argparse
import sys
import numpy as np
import os.path

# Initialize the parameters
confThreshold = 0.5  # Confidence threshold
nmsThreshold = 0.4  # Non-maximum suppression threshold
inpWidth = 512       # Width of network's input image
inpHeight = 512      # Height of network's input image

# Load classes
classesFile = "../model/classes.names"
with open(classesFile, 'rt') as f:
    classes = f.read().rstrip('\n').split('\n')

# Load the network
modelConfiguration = "../model/config/darknet-yolov3.cfg"
modelWeights = "/home/yossi/Documents/weights/lpr_yolo3/model.weights"
net = cv.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)

# Initialize multi tracker
multiTracker = cv.legacy.MultiTracker_create()

def getOutputsNames(net):
    layersNames = net.getLayerNames()
    return [layersNames[i - 1] for i in net.getUnconnectedOutLayers()]

def iou(box1, box2):
    """
    Calculate Intersection over Union (IoU) of two bounding boxes.
    """
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    
    xi1, yi1 = max(x1, x2), max(y1, y2)
    xi2, yi2 = min(x1+w1, x2+w2), min(y1+h1, y2+h2)
    
    inter_area = max(xi2 - xi1, 0) * max(yi2 - yi1, 0)
    
    box1_area = w1 * h1
    box2_area = w2 * h2
    
    union_area = box1_area + box2_area - inter_area
    
    return inter_area / union_area if union_area != 0 else 0

def drawPred(frame, classId, conf, left, top, right, bottom):
    # Draw a bounding box.
    cv.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 3)

def updateTrackersAndDetections(frame, outs, multiTracker):
    frameHeight, frameWidth = frame.shape[:2]
    classIds, confidences, boxes = [], [], []

    # Process detections
    for out in outs:
        for detection in out:
            scores = detection[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if confidence > confThreshold:
                center_x, center_y, width, height = (detection[0:4] * np.array([frameWidth, frameHeight, frameWidth, frameHeight])).astype(int)
                left, top = int(center_x - width / 2), int(center_y - height / 2)
                classIds.append(classId)
                confidences.append(float(confidence))
                boxes.append((left, top, width, height))  # Use tuple for consistency

    # Apply NMS
    indices = cv.dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThreshold)
    nms_boxes = [boxes[i] for i in indices]

    # Get updated tracker boxes
    success, tracked_boxes = multiTracker.update(frame)

    # Draw tracked boxes with updated positions
    for box in tracked_boxes:
        left, top, width, height = [int(v) for v in box]
        cv.rectangle(frame, (left, top), (left + width, top + height), (200,0,0), 2, 1)

    # Add new trackers for detected objects not already being tracked
    for i, box in enumerate(nms_boxes):
        # Check if this box overlaps significantly with any tracked box
        if not any(iou(box, tbox) > 0.5 for tbox in tracked_boxes):  # Assuming an iou() helper function
            tracker = cv.legacy.TrackerCSRT_create()
            multiTracker.add(tracker, frame, box)
            # Draw new detection
            drawPred(frame, classIds[i], confidences[i], box[0], box[1], box[0]+box[2], box[1]+box[3])


# Main loop
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Object Detection and Tracking with YOLO and OpenCV')
    parser.add_argument('--video', help='Path to video file.', default='../input.mp4')
    args = parser.parse_args()

    if args.video:
        cap = cv.VideoCapture(args.video)
    else:
        cap = cv.VideoCapture(0)

    while cv.waitKey(1) < 0:
        hasFrame, frame = cap.read()
        if not hasFrame:
            print("Done processing !!!")
            break

        # Prepare the frame for detection
        blob = cv.dnn.blobFromImage(frame, 1/255, (inpWidth, inpHeight), [0, 0, 0], 1, crop=False)
        net.setInput(blob)
        outs = net.forward(getOutputsNames(net))

        # Previously, tracker updates and detections were handled separately here
        # Now, we call the integrated function to manage both tasks
        updateTrackersAndDetections(frame, outs, multiTracker)

        # Display the frame
        cv.imshow('frame', frame)
        if cv.waitKey(1) & 0xFF == ord('q'):  # Allow breaking the loop with 'q'
            break

    cv.destroyAllWindows()
