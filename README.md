# Drowsiness Detection
 This is a Stem project

**Table of content** 

## Install library

<!-- all libray need in project -->
```cmd
pip install spatial
pip install imutils
pip install pygame
pip install dlib
pip install opencv
pip install opencv-python
```

## Working

<!-- what to do during the project -->

#### Import module

<!-- Import -->
```py
from scipy.spatial import distance
from imutils import face_utils
from pygame import mixer
import imutils
import dlib
import cv2
```

#### Import sound

<!-- Take sound file -->
```py
mixer.init()
mixer.music.load("./Sound/music.wav")
```

#### Make function to calculate the EAR

<!-- make calculated EAR code -->
```py
def eye_aspect_ratio(eye):
	A = distance.euclidean(eye[1], eye[5])
	B = distance.euclidean(eye[2], eye[4])
	C = distance.euclidean(eye[0], eye[3])
	ear = (A + B) / (2.0 * C)
	return ear
```
> EAR describle how the eye work (open or close)

#### Set up

**Caution index**
```py
thresh = 0.25
frame_check = 20
flag = 0
```
- set thresh EAR as <span style="color:red;">0.25</span>
- set thresh time as <span style="color:red;">20</span>
- set count thresh time as <span style="color:yellow;">0</span>
  
**Face landmark**
```py
detect = dlib.get_frontal_face_detector()
predict = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
```

**Define left and right eyes**
```py
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]
```

**Use camera**
```py
cap=cv2.VideoCapture(0)
```
> cap hiểu đơn giản là những gì thu được từ camera

#### Work

**Main function**
```py
while True:
	ret, frame=cap.read()
	frame = imutils.resize(frame, width=450)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	subjects = detect(gray, 0)
	for subject in subjects:
		shape = predict(gray, subject)
		shape = face_utils.shape_to_np(shape)
		leftEye = shape[lStart:lEnd]
		rightEye = shape[rStart:rEnd]
		leftEAR = eye_aspect_ratio(leftEye)
		rightEAR = eye_aspect_ratio(rightEye)
		ear = (leftEAR + rightEAR) / 2.0
		leftEyeHull = cv2.convexHull(leftEye)
		rightEyeHull = cv2.convexHull(rightEye)
		cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
		cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
		if ear < thresh:
			flag += 1
			print (flag)
			if flag >= frame_check:
				cv2.putText(frame, "****************ALERT!****************", (10, 30),
					cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
				cv2.putText(frame, "****************ALERT!****************", (10,325),
					cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
				mixer.music.play()
		else:
			flag = 0
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF
	if key == ord("q"):
		break
```
#### End program and close camera
```py
cv2.destroyAllWindows()
cap.release()
```

