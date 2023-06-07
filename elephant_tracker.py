import datetime
from ultralytics import YOLO
import cv2
from helper import create_video_writer
from deep_sort_realtime.deepsort_tracker import DeepSort
import numpy as np
import argparse
import os

def parse_args():
    parser = argparse.ArgumentParser(description='Run inference on test images')
    parser.add_argument('--source', required=True, help='Path to directory containing images')
    parser.add_argument('--output', required=True, help='Path to save the inference result')
    parser.add_argument('--weights',required=True, help='Path to checkpoint file')
    return parser.parse_args()
    
args = parse_args()

# Load the model
model = YOLO(args.weights)

#create output directory if it doesnot exist
os.makedirs(args.output, exist_ok=True)

# Parameters
CONFIDENCE_THRESHOLD = 0.8
GREEN = (0, 255, 0)
WHITE = (255, 255, 255)
threshold = 2.1  # Minimum displacement threshold for considering an object as "moving"

# initialize the video capture object
video_cap = cv2.VideoCapture(args.source)

# Get the video frame rate and dimensions
fps = int(video_cap.get(cv2.CAP_PROP_FPS))
frame_width = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# initialize the video writer object
writer = cv2.VideoWriter(os.path.join(args.output, "output.mp4"), cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

#tracker
tracker = DeepSort(max_age=50)

# Dictionary to store the trailing lines for each track
trailing_lines = {}



while True:
    start = datetime.datetime.now()

    ret, frame = video_cap.read()

    if not ret:
        break

    # run the YOLO model on the frame
    detections = model(frame)[0]

    # initialize the list of bounding boxes and confidences
    results = []

    ######################################
    # DETECTION
    ######################################

    # loop over the detections
    for data in detections.boxes.data.tolist():
        # extract the confidence (i.e., probability) associated with the prediction
        confidence = data[4]

        # filter out weak detections by ensuring the 
        # confidence is greater than the minimum confidence
        if float(confidence) < CONFIDENCE_THRESHOLD:
            continue

        # if the confidence is greater than the minimum confidence,
        # get the bounding box and the class id
        xmin, ymin, xmax, ymax = int(data[0]), int(data[1]), int(data[2]), int(data[3])
        class_id = int(data[5])
        # add the bounding box (x, y, w, h), confidence, class id, and center point to the results list
        results.append([[xmin, ymin, xmax - xmin, ymax - ymin], confidence, class_id, [(xmin + xmax) // 2, (ymin + ymax) // 2]])

    ######################################
    # TRACKING
    ######################################

    # update the tracker with the new detections
    tracks = tracker.update_tracks(results, frame=frame)

    # loop over the tracks
    for track in tracks:
        # if the track is not confirmed, ignore it
        if not track.is_confirmed():
            continue

        # get the track id, bounding box, and class id
        track_id = track.track_id
        ltrb = track.to_ltrb()
        xmin, ymin, xmax, ymax = int(ltrb[0]), int(ltrb[1]), int(ltrb[2]), int(ltrb[3])

        # Update the trailing line for the track
        if track_id not in trailing_lines:
            trailing_lines[track_id] = []
        center = (int((xmin + xmax) // 2), int((ymin + ymax) // 2))
        trailing_lines[track_id].append(center)

        # Check if the object is moving based on the displacement threshold
        if len(trailing_lines[track_id]) > 1:
            displacement = np.linalg.norm(np.array(trailing_lines[track_id][-1]) - np.array(trailing_lines[track_id][-2]))
            print(f'displacement is: {displacement}')
            if displacement > threshold:
                # Draw the trailing line
                for i in range(1, len(trailing_lines[track_id])):
                    cv2.line(frame, trailing_lines[track_id][i - 1], trailing_lines[track_id][i], GREEN, 2)


        # draw bounding box
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), GREEN, 2)
        
        # display class name and ID
        label = f"{track_id}"
        cv2.putText(frame, label, (xmin, ymin - 15), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 5)

    # end time to compute the fps
    end = datetime.datetime.now()
    # show the time it took to process 1 frame
    print(f"Time to process 1 frame: {(end - start).total_seconds() * 1000:.0f} milliseconds")
    # calculate the frame per second and draw it on the frame
    fps = f"FPS: {1 / (end - start).total_seconds():.2f}"
    cv2.putText(frame, fps, (50, 50),cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    cv2.putText(frame, "Model: Yolov8x", (10, frame_height - 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0),2)
    cv2.putText(frame, "Tracker: Deep_Sort_Realtime", (10, frame_height - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0),2)

    # show the frame to our screen
    cv2.imshow("Frame", frame)
    writer.write(frame)
    if cv2.waitKey(1) == ord("q"):
        break

video_cap.release()
writer.release()
cv2.destroyAllWindows()
