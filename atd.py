import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime

def findEncodings(list_of_images):
    encode_list = []
    for img in list_of_images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encode_list.append(encode)
    return encode_list


def markAttendance(person_name):
    with open('Attendance.csv', 'r+') as f:
        my_data_list = f.readlines()
        name_list = []
        for line in my_data_list:
            entry = line.split(',')
            name_list.append(entry[0])
        if person_name not in name_list:
            now = datetime.now()
            dt_string = now.strftime('%H:%M:%S')
            f.writelines(f'\n{person_name},{dt_string}')

