import cv2
import pandas as pd
from ultralytics import YOLO
from tracker import *
import cvzone
import numpy as np

model = YOLO('yolov8s.pt')

def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        point = [x, y]
        print(point)

cap = cv2.VideoCapture('input_video.mp4')

my_file = open("coco.txt", "r")
data = my_file.read()
class_list = data.split("\n")

count = 0

# area1=[(259,488),(281,499),(371,499),(303,466)]
area1 = [(269, 488), (281, 494), (361, 494), (303, 476)]
tracker = Tracker()

counter = []
output_frames = []

while True:
    ret, frame = cap.read()
    if not ret:
        break
    count += 1
    if count % 3 != 0:
        continue
    frame = cv2.resize(frame, (1020, 500))
    results = model.predict(frame)
    a = results[0].boxes.data
    px = pd.DataFrame(a).astype("float")
    list = []
    for index, row in px.iterrows():
        # coordinates,id and class
        x1 = int(row[0])
        y1 = int(row[1])
        x2 = int(row[2])
        y2 = int(row[3])
        d = int(row[5])
        c = class_list[d]
        if 'person' in c:
            # when person detect the find the coordinates
            list.append([x1, y1, x2, y2])
        # sent these coordinates to tracker
        bbox_idx = tracker.update(list)
        for id, rect in bbox_idx.items():
            x3, y3, x4, y4 = rect
            # cx, cy uesd for draw the polylines of circle
            cx = x3
            cy = y4
            result = cv2.pointPolygonTest(np.array(area1, np.int32), ((cx, cy)), False)
            if result >= 0:
                # cv2.circle(frame, (cx, cy), 6, (0, 255, 0), -1)
                cv2.rectangle(frame, (x3, y3), (x4, y4), (255, 0, 255), 2)
                cvzone.putTextRect(frame, f'{id}', (x3, y3), 1, 1)
                if counter.count(id) == 0:
                    counter.append(id)

        # cv2.polylines(frame, [np.array(area1, np.int32)], True, (255, 0, 0), 2)
        person_count = len(counter)
        cvzone.putTextRect(frame, f'Counter:{person_count}', (20, 30), 2, 2)
        output_frames.append(frame.copy())

    cv2.imshow("Object Detect", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Release video capture
cap.release()
cv2.destroyAllWindows()

# Save frames as a video
height, width, _ = output_frames[0].shape
out = cv2.VideoWriter('output_video.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, (width, height))
for frame in output_frames:
    out.write(frame)
out.release()
