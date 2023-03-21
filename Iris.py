import mediapipe as mp
import numpy as np
import cv2 as cv
"""FACEMESH_LEFT_IRIS = frozenset([(474, 475), 
                                   (475, 476), 
                                   (476, 477),
                                   (477, 474)])"""
"""""FACEMESH_RIGHT_IRIS = frozenset([(469, 470), 
                                      (470, 471), 
                                      (471, 472),
                                      (472, 469)])"""
mp_drawing=mp.solutions.drawing_utils
mp_face=mp.solutions.face_mesh
class face_detection:
    def __init__(self,static_image=False,refine_landmarks=False,maxfaces=1,confdetection=.5,conftracker=.5):
        self.static_image=static_image
        self.maxfaces=maxfaces
        self.ref_landmarks=refine_landmarks
        self.confdetection=confdetection
        self.conftracker=conftracker
        self.face_dir=mp_face
        self.drawing_utils=mp_drawing
        self.face=self.face_dir.FaceMesh(self.static_image,self.maxfaces,self.ref_landmarks,self.confdetection,self.conftracker)
    def draw(self,frame):
        imgcolor=cv.cvtColor(frame,cv.COLOR_BGR2RGB)
        self.resultados=self.face.process(imgcolor)
        self.landmarks=self.resultados.face_landmarks
        #self.landmarks=self.resultados.multi_face_landmarks
        """"if self.landmarks:
            for landmark in self.landmarks:
                print(landmark)"""
        self.drawing_utils.draw_landmarks(frame,self.landmarks,self.face_dir.FACEMESH_TESSELATION)
        return frame

#testing area
detection=face_detection()
cap=cv.VideoCapture(-1)
while cap.isOpened():
    ret,frame=cap.read()
    frame=detection.draw(frame)
    cv.imshow("Is0k?",frame)
    if cv.waitKey(10) & 0xFF==ord("q"):
        break
cap.release()
cv.destroyAllWindows()

