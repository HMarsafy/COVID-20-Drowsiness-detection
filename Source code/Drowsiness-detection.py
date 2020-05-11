# import the necessary packages
import cv2
from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils import face_utils
from threading import Thread
import playsound
import imutils
import time
import dlib

#flag to stop the alarm of the closing eye.
stop_thread = False
#flag to stop the alarm of face disappearence.
stop_thread_face = False
#flag to face detection.
face_detected = False


# function that is responsible of start the "stop driving" alarm.
def stop_driving(path):
	for i in range(0,4):
		playsound.playsound(path)
		


# function that is responsible of start the "face disappearence" alarm.
def face_disappear(path):
    while True:
        playsound.playsound(path)
        global stop_thread_face
        if stop_thread_face:
            break

# function that is responsible of start the "eye closing" alarm
def alarm_sound(path):
    while True:
        playsound.playsound(path)
        global stop_thread
        if stop_thread:
            break

# function that is responsible of caculating eye aspect ratio .
def eye_aspect_ratio(eye):
    # compute the euclidean distances between the two sets of
    # vertical eye landmarks (x, y)-coordinates
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])

    # compute the euclidean distance between the horizontal
    # eye landmark (x, y)-coordinates
    C = dist.euclidean(eye[0], eye[3])

    # compute the eye aspect ratio
    ear = (A + B) / (2.0 * C)

    # return the eye aspect ratio
    return ear
# eye aspect ratio threshold
EYE_AR_THRESH = 0.3
# number of Consecutive frames to start the "eye closing" alarm
EYE_AR_CONSEC_FRAMES = 30
# number of Consecutive frames to start the "face disappearence" alarm
FACE_CONSEC_FRAMES = 60

# initialize the frame counter of  Consecutive frames as well as a boolean used to
# indicate if the alarm is going off
COUNTER = 0
# initialize the frame counter of  Consecutive frames as well as a boolean used to
# indicate if the alarm of face disappearence is going off
face_counter = 0
# initialize the frame counter of Consecutive closing eye times as well as a boolean used to
# indicate if the alarm is going off
stop_driving_counter=0
#flag to know the situation of the alarm right now.
ALARM_ON = False
#flag to know the situation of the alarm rught now.
FACE_DISAPPEAR_ALARM = False

# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

# grab the indexes of the facial landmarks for the left and
# right eye, respectively
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

# start the video stream thread
vs = VideoStream(0).start()
time.sleep(1.0)


# loop over frames from the video stream
while True:
    # grab the frame from the threaded video file stream, resize
    # it, and convert it to grayscale
    # channels.
#check the stop_driving_counter value
    if stop_driving_counter == 3:
	 #create a thread to turn on the alarm when the stop_driving_counter is equal to 3
    	 t1 = Thread(target=stop_driving, args=('stop_driving.mp3',))
         t1.deamon = True
         t1.start()
	 # reset the counter to start from zero again
         stop_driving_counter = 0  
     # take frame
    frame = vs.read()
    frame = imutils.resize(frame, width=900)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # detect faces in the grayscale frame
    rects = detector(gray, 0)
    #if detect faces in the frame.
    if len(rects) != 0:
        face_detected = True
        face_counter = 0
        FACE_DISAPPEAR_ALARM = False
        stop_thread_face = True
    #if there is no faces in the frame but it was detected before.
    elif len(rects) == 0 and face_detected==True:
        face_counter += 1
	#if the  Consecutive frames is greater than 60
        if face_counter > FACE_CONSEC_FRAMES:
            if not FACE_DISAPPEAR_ALARM:
                stop_thread_face = False
                FACE_DISAPPEAR_ALARM = True
		##create a thread to turn on the alarm when the the face_counter is greater than 60
                t = Thread(target=face_disappear, args=('alarm.wav',))
                t.deamon = True
                t.start()

    # loop over the face detections
    for rect in rects:
        # determine the facial landmarks for the face region, then
        # convert the facial landmark (x, y)-coordinates to a NumPy
        # array
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        # extract the left and right eye coordinates, then use the
        # coordinates to compute the eye aspect ratio for both eyes
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)

        # check to see if the eye aspect ratio is below the blink
        # threshold, and if so, increment the blink frame counter
        if leftEAR < EYE_AR_THRESH and rightEAR < EYE_AR_THRESH:

            COUNTER += 1

            # if the eyes were closed for a sufficient number of
            # then sound the alarm
            if COUNTER >= EYE_AR_CONSEC_FRAMES:
                # if the alarm is not on, turn it on
                if not ALARM_ON:
                    stop_thread = False
                    ALARM_ON = True
                    t = Thread(target=alarm_sound, args=('alarm.wav',))
                    stop_driving_counter+=1
                    t.deamon = True
                    t.start()

        # otherwise, the eye aspect ratio is not below the blink
        # threshold, so reset the counter and alarm
        else:
            COUNTER = 0
            ALARM_ON = False
            stop_thread = True

    # show the frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
