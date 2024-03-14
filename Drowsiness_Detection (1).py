from scipy.spatial import distance
from imutils import face_utils
from pygame import mixer
import imutils
import dlib
import cv2

# define mixer
mixer.init()
mixer.music.load("./Sound/music.wav")

# Calulate EAR
def eye_aspect_ratio(eye):
	A = distance.euclidean(eye[1], eye[5])
	B = distance.euclidean(eye[2], eye[4])
	C = distance.euclidean(eye[0], eye[3])
	ear = (A + B) / (2.0 * C)
	return ear

# define caution index
thresh = 0.25
frame_check = 20

# make face landmark
detect = dlib.get_frontal_face_detector()
predict = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Define left and right eyes
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]

# Use camera
cap=cv2.VideoCapture(0)

# define in threshed time
flag=0

# main function
while True:
	ret, frame=cap.read()
	# set frame
	frame = imutils.resize(frame, width=450)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	subjects = detect(gray, 0)
	for subject in subjects:
		shape = predict(gray, subject)
		shape = face_utils.shape_to_np(shape)
		# define two eyes
		leftEye = shape[lStart:lEnd]
		rightEye = shape[rStart:rEnd]
		# compute EAR
		leftEAR = eye_aspect_ratio(leftEye)
		rightEAR = eye_aspect_ratio(rightEye)
		# average EAR of two eyes
		ear = (leftEAR + rightEAR) / 2.0
		# Eyes tracker draw
		leftEyeHull = cv2.convexHull(leftEye)
		rightEyeHull = cv2.convexHull(rightEye)
		cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
		cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
		# Check EAR
		if ear < thresh:
			flag += 1 #flag = flag + 1
			print (flag)
			if flag >= frame_check:
				# Caution on the screen
				cv2.putText(frame, "****************ALERT!****************", (10, 30),
					cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
				cv2.putText(frame, "****************ALERT!****************", (10,325),
					cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
				# play the Caution sound
				mixer.music.play()
		else:
			# Set time to 0
			flag = 0
	# Show Frame
	cv2.imshow("Frame", frame)
	# bind key to exit
	key = cv2.waitKey(1) & 0xFF
	if key == ord("q"):
		break
# End work
cv2.destroyAllWindows()
# Stop the camera
cap.release()