import cv2
import os
import numpy as np
from face_recognition import face_locations
from face_recognition import face_encodings, face_distance, load_image_file

known_encodings = []
known_names = []

for file in os.listdir('known_persons'):
    name = file[:-4]
    image = load_image_file('known_persons/' + file)
    face_encoding = face_encodings(image)[0]
    known_encodings.append(face_encoding)
    known_names.append(name)


cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    rgb_frame = frame[:, :, ::-1]
    locations = face_locations(rgb_frame)
    encodings = face_encodings(rgb_frame, locations)

    for location, encoding in zip(locations, encodings):
        top, right, bottom, left = location

        frame = cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0))
        distances = face_distance(known_encodings, encoding)
        distance = np.min(distances)
        min_index = np.argmin(distances)
        if distance > 0.65:
            name = 'Unknown'
        else:
            name = known_names[min_index] + " " + str(distance)
        
        frame = cv2.putText(frame, name, (left, bottom - 10), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0))


    cv2.imshow('video', frame)