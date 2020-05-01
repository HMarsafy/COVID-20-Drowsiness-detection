# loop over the (x, y)-coordinates for the facial landmarks
# and draw them on the image
for (x, y) in shape:
  cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)  
	# show the frame
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF
