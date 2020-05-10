def backend():
    # define two constants, one for the eye aspect ratio to indicate
    # blink and then a second constant for the number of consecutive
    # frames the eye must be below the threshold for to set off the
    # alarm
    EYE_AR_THRESH = 0.3
    EYE_AR_CONSEC_FRAMES = 15
    FACE_CONSEC_FRAMES = 30

    # initialize the frame counter as well as a boolean used to
    # indicate if the alarm is going off
    COUNTER = 0
    face_counter = 0
    stop_driving_counter = 0
    ALARM_ON = False
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

    global exit_flag
    while not exit_flag:
        # grab the frame from the threaded video file stream, resize
        # it, and convert it to grayscale
        # channels)
        ########################################
        global stop_thread_face
        global stop_thread
        global face_detected
        if not hold:
            if stop_driving_counter == 3:
                t1 = Thread(target=stop_driving, args=('stop_driving.mp3',))
                t1.deamon = True
                t1.start()
                stop_driving_counter = 0
                ########################################
            frame = vs.read()
            frame = imutils.resize(frame, width=900)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # detect faces in the grayscale frame
            rects = detector(gray, 0)
            if len(rects) != 0:
                face_detected = True
                face_counter = 0
                FACE_DISAPPEAR_ALARM = False
                stop_thread_face = True
            elif len(rects) == 0 and face_detected == True:
                face_counter += 1
                if face_counter > FACE_CONSEC_FRAMES:
                    if not FACE_DISAPPEAR_ALARM:
                        stop_thread_face = False
                        FACE_DISAPPEAR_ALARM = True
                        t2 = Thread(target=face_disappear, args=('alarm.mp3',))
                        t2.deamon = True
                        t2.start()

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
                            t3 = Thread(target=alarm_sound, args=('alarm.mp3',))
                            stop_driving_counter += 1
                            t3.deamon = True
                            t3.start()

                # otherwise, the eye aspect ratio is not below the blink
                # threshold, so reset the counter and alarm
                else:
                    COUNTER = 0
                    ALARM_ON = False
                    stop_thread = True

            # show the frame
            # cv2.imshow("Frame", frame)
            key = cv2.waitKey(1) & 0xFF

            # if the `q` key was pressed, break from the loop
            if key == ord("q"):
                break
        else:
            stop_thread_face = True
            stop_thread = True
            face_detected = False
            ALARM_ON = False
    cv2.destroyAllWindows()
    vs.stop()
