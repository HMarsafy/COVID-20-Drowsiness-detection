# loop over the frames from the video stream
while True:
	# grab the frame from the threaded video stream.
	frame = video.read()
        # resize it to have a maximum width of 400 pixels, and convert it to gray scale.
	frame = imutils.resize(frame, width=400)
 	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	# detect faces in the grayscale frame
	faces_in_video = detector(gray, 0)
for face in faces_in_video:
	# determine the facial landmarks for the face region, then
	# convert the facial landmark (x, y)-coordinates to a NumPy
	# array
	shape = predictor(gray, face)
	shape = face_utils.shape_to_np(shape)
