import cv2
import os
import numpy as np
from face_recognition import face_locations

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    rgb_frame = frame[:, :, ::-1]
    locations = face_locations(rgb_frame)

    for location in locations:
        top, right, bottom, left = location

        frame = cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0))

    cv2.imshow('video', frame)