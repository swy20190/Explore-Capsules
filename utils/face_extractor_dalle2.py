import os
import cv2
import dlib
import argparse
import numpy as np

hog_face_detector = dlib.get_frontal_face_detector()

image_folder = "../dalle2"
dst_folder = "../dalle2_faces"
crop_area = 80
detector = dlib.get_frontal_face_detector()
cnt = 1
for img_path in os.listdir(image_folder):

    #   t1= time.perf_counter()

    img = cv2.imread(os.path.join(image_folder, img_path))

    faces_hog = hog_face_detector(img, 1)

    for face in faces_hog:
        x = face.left()
        y = face.top()
        w = face.right() - x
        h = face.bottom() - y

        # draw box over face
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        face_img = img[y:y + h, x:x + w]

        if face_img.shape[0] >= 70 and face_img.shape[1] >= 70:
            print(cnt)
            face_img_norm = cv2.resize(face_img, (224, 224))

            cv2.imwrite(os.path.join(dst_folder, f"face{cnt:03}.jpg"), face_img_norm)
            cnt += 1


