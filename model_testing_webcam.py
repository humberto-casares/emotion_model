import cv2
import os
import numpy as np
import dlib
import imutils
from keras.models import load_model
from mtcnn.mtcnn import MTCNN

window_size = 24
window_step = 6
height = 48
width = 48


predictor = dlib.shape_predictor('./api_code/shape_predictor_68_face_landmarks.dat')
model = load_model("./api_code/best_model.h5")
face_detector = MTCNN()
face_rects = [dlib.rectangle(left=1, top=1, right=47, bottom=47)]
emotion = ["negative", "neutral", "positive"]


def get_landmarks(image, rects):
    # this function have been copied from http://bit.ly/2cj7Fpq
    if len(rects) > 1:
        raise BaseException("TooManyFaces")
    if len(rects) == 0:
        raise BaseException("NoFaces")
    return np.matrix([[p.x, p.y] for p in predictor(image, rects[0]).parts()])


def sliding_hog_windows(image):
    hog_windows = []
    for y in range(0, height, window_step):
        for x in range(0, width, window_step):
            window = image[y:y+window_size, x:x+window_size]
            hog_windows.extend(hog(window, orientations=8, pixels_per_cell=(8, 8),
                                            cells_per_block=(1, 1), visualise=False))
    return hog_windows



cap = cap = cv2.VideoCapture(0) 
# frame = cv2.imread("/home/dipak/Downloads/MISSILEMAN.jpeg")      
            
while(cap.isOpened()): 
    ret, frame = cap.read()
    frame = imutils.resize(frame, width=600)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detect_faces(frame)

    if ret == True: 

        for _ in faces:
            x, y, w, h = faces[0]['box']

            if w < 30 and h < 30:
                continue


            f = frame[y:y+h, x:x+w].copy()
            f = cv2.resize(f, (48, 48))

            face = gray[y:y+h, x:x+w].copy()
            face = cv2.resize(face, (48, 48)).astype("float32")

            face /= 255.0
            face = np.reshape(face, (1, 48, 48, 1))

            landmarks = get_landmarks(f, face_rects)
            landmarks = np.array(landmarks, dtype='float32')
            landmarks = landmarks.flatten()
            landmarks = landmarks.reshape(1, -1)

            pred = model.predict([face, landmarks])[0]


            j = np.argmax(pred)
            confidence = pred[j] * 100
            pred = emotion[j]

            text = "{} ({:.2f}%)".format(pred, confidence)
            cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)

            cv2.imshow("Emotion_Detection", frame)
  
   
        # Press Q on keyboard to  exit 
        if cv2.waitKey(25) & 0xFF == ord('q'): 
            break
   
    # Break the loop 
    else:  
        break
   

cap.release() 
   
cv2.destroyAllWindows() 
