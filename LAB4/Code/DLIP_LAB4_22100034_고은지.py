#########################################################
# Author : Eunji Ko
# Date   : 2024.6.6
# Class  : Deep Learning and Image Processing
# LAB    : LAB4, Object Detection
#########################################################

import cv2 as cv
from ultralytics import YOLO
import numpy as np
import matplotlib.pyplot as plt

plt.ioff()  

# Load the YOLOv8 model
model = YOLO('yolov8n.pt')

# Open the video file
video_path = 'Code/DLIP_parking_test_video.mp4'
cap = cv.VideoCapture(video_path)

# If not success, exit the program
if not cap.isOpened():
    print('Cannot open video')

# Define ROI coordinates and size
x, y, width, height = 0, 240, 1280, 180

# Parking Points
PP = np.zeros((13, 4))

PP[:, 1] = 320
PP[:, 3] = 430

initial_values = [
    [60, 160],
    [160, 260],
    [260, 360],
    [360, 455],
    [455, 547],
    [547, 635],
    [635, 725],
    [725, 815],
    [815, 905],
    [905, 995],
    [995, 1090],
    [1090, 1190],
    [1190, 1280]
]

for i, (val1, val2) in enumerate(initial_values):
    PP[i, 0] = val1
    PP[i, 2] = val2

# Prevent the Duplicated bbox
def calculate_iou(box1, box2):
    x1, y1, x2, y2 = box1
    x3, y3, x4, y4 = box2

    xi1 = max(x1, x3)
    yi1 = max(y1, y3)
    xi2 = min(x2, x4)
    yi2 = min(y2, y4)

    inter_width = max(0, xi2 - xi1 + 1)
    inter_height = max(0, yi2 - yi1 + 1)
    inter_area = inter_width * inter_height

    box1_area = (x2 - x1 + 1) * (y2 - y1 + 1)
    box2_area = (x4 - x3 + 1) * (y4 - y3 + 1)

    union_area = box1_area + box2_area - inter_area

    iou = inter_area / union_area
    return iou

# draw the cars
def drawcar(frame, results):
    carnum = 0
    drawn_boxes = []

    for result in results:
        for box in result.boxes:
            if (int(box.cls) in [2.0, 5.0, 7.0]):
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                is_duplicate = False
                for drawn_box in drawn_boxes:
                    iou = calculate_iou((x1, y1, x2, y2), drawn_box)
                    if iou > 0.2:           # if IoU is more than 0.2, duplicated
                        is_duplicate = True
                        break

                if not is_duplicate:
                    carnum += 1
                    cv.rectangle(frame, (x1, y1 + y), (x2, y2 + y), (0, 0, 255), 2)
                    cv.putText(frame, 'car', (x1, y1 + y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                    drawn_boxes.append((x1, y1, x2, y2))

    return carnum

# checking available parking spaces with the points of bbox and parking points(PP)
def ParkingSpace(results, PP):
    PA = [0] * 13
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            x_avg = (x1 + x2) / 2
            y_avg = (y1 + y + y2 + y) / 2
            for i in range(13):
                if x_avg > PP[i][0] and x_avg < PP[i][2] and y_avg > PP[i][1] and y_avg < PP[i][3]:
                    PA[i] = 1
    return PA

frame_number = 0

with open("counting_result.txt", "w") as f:
    # Loop through the video frames
    while cap.isOpened():
        success, frame = cap.read()
        count = 0
        if success:
            frame_height, frame_width, channels = frame.shape
            roi = frame[y:y+height, x:x+width]
            results = model(roi)
            boxes = results[0].boxes

            carnum = drawcar(frame, results) # the number of cars
            PA = ParkingSpace(results, PP)   # information of available parking spaces

            for i in range(13):
                if PA[i] == 0:               # if there is no cars in the each parking space
                    count += 1               # avavilable parking space += 1
            
            # Print count & carnum
            cv.putText(frame, f'Available Parking: {count}', (900, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv.putText(frame, f'Number of Car: {carnum}', (900, 100), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # save the frame number and count to text file
            f.write(f'{frame_number},{count}\n')

            frame_number += 1

            cv.imshow("Result", frame)
            if cv.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            break

cap.release()
cv.destroyAllWindows()
