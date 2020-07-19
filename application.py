from datetime import time

import cv2
import os
import numpy as np
import face_recognition
from flask import Flask, jsonify, request
from flask_restful import Resource, Api
# creating the flask app
from atd import findEncodings, markAttendance

app = Flask(__name__)
# creating an API object
api = Api(app)


class GetImage(Resource):

    def get(self):
        path = 'ImagesAttendance'
        images = []
        result = None
        classNames = []
        myList = os.listdir(path)
        print(myList)
        for cl in myList:
            curImg = cv2.imread(f'{path}/{cl}')
            images.append(curImg)
            classNames.append(os.path.splitext(cl)[0])
        print(classNames)
        encodeListKnown = findEncodings(images)
        print('Encoding Complete')
        cap = cv2.VideoCapture(0)
        while True:
            success, img = cap.read()
            # img = captureScreen()
            imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
            imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

            facesCurFrame = face_recognition.face_locations(imgS)
            encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

            for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
                matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
                faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
                # print(faceDis)
                matchIndex = np.argmin(faceDis)

                if matches[matchIndex]:
                    name = classNames[matchIndex]
                    # print(name)
                    y1, x2, y2, x1 = faceLoc
                    y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
                    cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
                    result = name
                    markAttendance(name)

            cv2.imshow('Webcam', img)
            cv2.waitKey(1)
            if result in classNames:
                break
        return jsonify({'result': 'Person name is {0}'.format(result)})

class Test(Resource):
    def get(self):
        return "Hi Test"

api.add_resource(GetImage,'/get/')
api.add_resource(Test,'/')
