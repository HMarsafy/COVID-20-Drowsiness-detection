# we need to initialize dlib’s "HOG + Linear SVM-based face detector" and then load the facial landmark.
# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])

# 1) so we will detect the face first using detector.
# 2) then detect all the face landmarks using predictor that will put each facial landamark in a (x,y) coordinate form
