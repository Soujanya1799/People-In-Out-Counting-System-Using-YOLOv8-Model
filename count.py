import cv2
import argparse
import sys
import numpy as np
from ultralytics import YOLO
from centroidtracker import CentroidTracker
from trackableobject import TrackableObject
import os

# Initialize the parameters
confThreshold = 0.2  # Confidence threshold
nmsThreshold = 0.4   # Non-maximum suppression threshold

parser = argparse.ArgumentParser(description='Object Detection using YOLOv8 in OpenCV')
parser.add_argument('--video', default='test.mp4', help='Path to video file.')
args = parser.parse_args()

# Load the YOLOv8 model
model = YOLO("yolov8s.pt")  # Replace with the path to your YOLOv8 model

# instantiate our centroid tracker, then initialize a dictionary to
# map each unique object ID to a TrackableObject
ct = CentroidTracker(maxDisappeared=40, maxDistance=50)
trackableObjects = {}

# initialize the total number of frames processed thus far, along
# with the total number of objects that have moved either up or down
totalDown = 0
totalUp = 0

def counting(objects, frame):
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]

    global totalDown
    global totalUp

    # loop over the tracked objects
    for (objectID, centroid) in objects.items():
        # check to see if a trackable object exists for the current object ID
        to = trackableObjects.get(objectID, None)

        # if there is no existing trackable object, create one
        if to is None:
            to = TrackableObject(objectID, centroid)

        # otherwise, there is a trackable object so we can utilize it to determine direction
        else:
            # the difference between the y-coordinate of the *current* centroid and the mean of *previous* centroids will tell
            # us in which direction the object is moving (negative for 'up' and positive for 'down')
            y = [c[1] for c in to.centroids]
            direction = centroid[1] - np.mean(y)
            to.centroids.append(centroid)

            # check to see if the object has been counted or not
            if not to.counted:
                # if the direction is negative (indicating the object is moving up) AND the centroid is above the center line, count the object
                if direction < 0 and centroid[1] in range(frameHeight // 2 - 30, frameHeight // 2 + 30):
                    totalUp += 1
                    to.counted = True

                # if the direction is positive (indicating the object is moving down) AND the centroid is below the center line, count the object
                elif direction > 0 and centroid[1] in range(frameHeight // 2 - 30, frameHeight // 2 + 30):
                    totalDown += 1
                    to.counted = True

        # store the trackable object in our dictionary
        trackableObjects[objectID] = to

    # Print the counts to the console
    print(f"Up: {totalUp}, Down: {totalDown}")

# Process inputs
outputFile = "yolo_out_py.avi"

if args.video:
    # Open the video file
    if not os.path.isfile(args.video):
        print(f"Input video file {args.video} doesn't exist")
        sys.exit(1)
    cap = cv2.VideoCapture(args.video)
    outputFile = args.video[:-4] + '_output.avi'
else:
    # Webcam input
    cap = cv2.VideoCapture(0)

# Get the video writer initialized to save the output video
vid_writer = cv2.VideoWriter(outputFile, cv2.VideoWriter_fourcc(*'MJPG'), 30,
                              (round(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

while cap.isOpened():
    # get frame from the video
    hasFrame, frame = cap.read()

    # Skip the rest of the loop if frame is None
    if not hasFrame:
        break

    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]
    cv2.line(frame, (0, frameHeight // 2), (frameWidth, frameHeight // 2), (0, 255, 255), 2)

    # Perform object detection using YOLOv8
    results = model.predict(source=frame, conf=confThreshold, iou=nmsThreshold, classes=[0])  # Detect only 'person' class

    # Update object centroids and count objects
    rects = []
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            rects.append((x1, y1, x2, y2))
            # Draw bounding box on the frame
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    objects = ct.update(rects)
    counting(objects, frame)

    # Draw up and down counts on the frame
    from cv2 import putText  # Import putText from cv2
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 0.5
    fontColor = (0, 255, 0)
    lineType = 1
    y0, dy = 20, 15  # Adjust these values to position the text

    putText(frame, f'Up: {totalUp}', (10, y0), font, fontScale, fontColor, lineType)
    putText(frame, f'Down: {totalDown}', (10, y0 + dy), font, fontScale, fontColor, lineType)

    # Write the frame with the detection boxes
    vid_writer.write(frame.astype(np.uint8))

# Release device
cap.release()
vid_writer.release()

print("Done processing !!!")
print(f"Output file is stored as {outputFile}")