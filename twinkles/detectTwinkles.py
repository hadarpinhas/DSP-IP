import cv2
import numpy as np
from collections import deque

class TwinkleTracker:
    def __init__(self):
        self.previous_spots = {}

    def update_spots(self, current_frame, buffer):
        # Detect spots with significant changes in brightness over the last 10 frames.
        bright_spots = detect_spots(current_frame, buffer)

        # Update the tracker with the current bright spots
        current_spots = {}
        for cnt in bright_spots:
            x, y, w, h = cv2.boundingRect(cnt)
            center = (x + w // 2, y + h // 2)
            size = w * h
            # ... (rest of the update logic as previously defined)

    def get_twinkles(self):
        # ... (method to get twinkles as previously defined)
        return []

    # cv2.imshow('th', thresh)
    # cv2.waitKey(0)


def find_transformation_matrix(buffer):
    """
    Compute the transformation matrix to stabilize the frames.
    Assumes the last frame in the buffer is the reference frame.
    """
    # Convert buffer to a list for slicing
    buffer_list = list(buffer)

    # Ensure there is more than one frame to compare
    if len(buffer_list) < 2:
        return np.eye(3, dtype=np.float32)  # Return an identity matrix

    # Use the last frame in the buffer as the reference frame
    reference_frame = buffer_list[-1]
    
    # Initialize feature detector - ORB is a good balance between speed and effectiveness
    orb = cv2.ORB_create()

    # Detect and compute keypoints and descriptors in the reference frame
    keypoints_ref, descriptors_ref = orb.detectAndCompute(reference_frame, None)
    
    # Initialize variables for transformation matrix
    max_matches = 0
    best_matrix = None

    # Iterate over the frames in the buffer to find the best transformation
    for frame in buffer_list[:-1]:  # Skip the last frame (reference frame)
        keypoints_frame, descriptors_frame = orb.detectAndCompute(frame, None)
        
        # Feature matching
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = matcher.match(descriptors_frame, descriptors_ref)
        
        # Only proceed with enough matches
        if len(matches) > max_matches and len(matches) > 10:  # Arbitrary minimum number of matches
            max_matches = len(matches)
            
            # Extract location of good matches
            points_frame = np.float32([keypoints_frame[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
            points_ref = np.float32([keypoints_ref[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
            
            # Compute the transformation matrix using RANSAC for robustness
            matrix, _ = cv2.findHomography(points_frame, points_ref, cv2.RANSAC, 5.0)
            
            if matrix is not None:
                best_matrix = matrix

    # If no good matrix was found, return identity matrix
    if best_matrix is None:
        print(" best_matrix is None")
        best_matrix = np.eye(3, dtype=np.float32)

    return best_matrix

numMedianImgs = 5
# Initialize a buffer (queue) for the last 10 frames
last_frames_buffer = deque(maxlen=numMedianImgs)

def detect_spots(current_frame, buffer):
    # Convert the current frame to grayscale
    current_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blurring with two different kernel sizes
    blur1 = cv2.GaussianBlur(current_gray, (5, 5), 0)
    blur2 = cv2.GaussianBlur(current_gray, (9, 9), 0)

    # Compute the Difference of Gaussian
    dog = cv2.subtract(blur1, blur2)
    dog_normalized = cv2.normalize(dog, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    _, thresh1 = cv2.threshold(dog_normalized, 240, 255, cv2.THRESH_BINARY)

    cv2.imshow('thresh1', thresh1)

    kernel = np.ones((11, 11), np.uint8)
    dog_normalized = cv2.morphologyEx(thresh1, cv2.MORPH_CLOSE, kernel)

    cv2.imshow('dog_normalized', dog_normalized)
    # current_gray = dog_normalized

    # cv2.waitKey(0) 
    
    # Update the buffer with the current grayscale frame
    buffer.append(current_gray)
    
    # If we don't yet have 10 frames, return as we cannot compute the median background yet
    if len(buffer) < numMedianImgs:
        print(len(buffer))
        return []
    
    # Find the transformation matrix to align the current frame to the reference frame
    transformation_matrix = find_transformation_matrix(buffer)
    
    # Apply the transformation to stabilize the frame
    stabilized_frame = cv2.warpPerspective(current_gray, transformation_matrix, (current_gray.shape[1], current_gray.shape[0]))
    
    # Compute the median of the frames in the buffer to get the background
    # Ensure you apply the same transformation to each frame in the buffer
    stabilized_frames = [cv2.warpPerspective(f, transformation_matrix, (f.shape[1], f.shape[0])) for f in buffer]
    median_frame = np.median(np.array(stabilized_frames), axis=0).astype(dtype=np.uint8)
    
    # Calculate the absolute difference between the stabilized current frame and the median background
    frame_diff = cv2.absdiff(stabilized_frame, median_frame)
    
    # Threshold the difference to identify significant changes
    _, thresh = cv2.threshold(frame_diff, 70, 255, cv2.THRESH_BINARY)
    cv2.imshow('th', thresh)
    cv2.waitKey(0)    
    # Apply morphological opening to clean up small spots and noise
    kernel = np.ones((3, 3), np.uint8)
    opened = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    # cv2.imshow('opened', opened)
    # cv2.waitKey(0)        
    # Find contours
    contours, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter contours based on the spot size and brightness contrast
    bright_spots = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 30:  # Smaller than 30 pixels in area
            mask = np.zeros_like(current_gray)
            cv2.drawContours(mask, [cnt], -1, 255, -1)
            mean_val = cv2.mean(current_gray, mask=mask)[0]
            surrounding = cv2.mean(current_gray, cv2.bitwise_not(mask))[0]
            if mean_val > surrounding + 50:  # Adjust this value for required contrast
                bright_spots.append(cnt)
    
    return contours

def draw_twinkles(frame, twinkles):
    """
    Draw circles on twinkles.
    """
    for center in twinkles:
        cv2.circle(frame, center, 10, (0, 255, 0), 2)
    return frame

# Initialize the twinkle tracker
tracker = TwinkleTracker()

# Load the video
cap = cv2.VideoCapture('5.m4v')


imgH = 300
imgW = 300
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    # cv2.imshow('th', thresh)
    # cv2.waitKey(0)
    frame = frame[frame.shape[0]//2-imgH:frame.shape[0]//2+imgH, frame.shape[1]//2-imgW: frame.shape[1]//2+imgW]

    # Update tracker with the current frame and buffer
    tracker.update_spots(frame, last_frames_buffer)

    # Get twinkles (spots that have grown and then started to shrink)
    twinkles = tracker.get_twinkles()

    # Draw twinkles on the frame
    twinkle_frame = draw_twinkles(frame, twinkles)

    # Display the frame
    cv2.imshow('Twinkle Detection', twinkle_frame)

    # Break the loop on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all frames
cap.release()
cv2.destroyAllWindows()
