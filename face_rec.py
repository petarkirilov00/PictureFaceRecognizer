import face_recognition as face_rec
import os
import cv2
import face_recognition
import numpy as np
from time import sleep


def encodedFaces():
    encoded = {}

    for dirpath, dnames, fnames in os.walk("./pictures"):
        for fn in fnames:
            if fn.endswith(".jpg") or fn.endswith(".png"):
                face = face_rec.load_image_file("pictures/" + fn)
                encoding = face_rec.face_encodings(face)[0]
                encoded[fn.split(".")[0]] = encoding

    return encoded


def unknownFaces(img):
    face = face_rec.load_image_file("pictures/" + img)
    encoding = face_rec.face_encodings(face)[0]

    return encoding


def face(im):

    faces = encodedFaces()
    faces_encoded = list(faces.values())
    known_face_names = list(faces.keys())

    img = cv2.imread(im, 1)

    face_locations = face_recognition.face_locations(img)
    unknown_face_encodings = face_recognition.face_encodings(img, face_locations)

    face_names = []

    for face_encoding in unknown_face_encodings:
        matches = face_recognition.compare_faces(faces_encoded, face_encoding)
        name = "Unknown"

        face_distances = face_recognition.face_distance(faces_encoded, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]

        face_names.append(name)

        for (top, right, bottom, left), name in zip(face_locations, face_names):
            cv2.rectangle(img, (left-20, top-20), (right+20, bottom+20), (255, 0, 0), 2)
            cv2.rectangle(img, (left-20, bottom -15), (right+20, bottom+20), (255, 0, 0), cv2.FILLED)

            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(img, name, (left -20, bottom + 15), font, 1.0, (255, 255, 255), 2)


    while True:
        cv2.imshow('Video', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            return face_names


print(face("try.jpg"))


