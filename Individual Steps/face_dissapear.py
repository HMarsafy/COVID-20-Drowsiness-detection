#function to start the alarm when the face dissapears and stop it when the flag changes.
def face_disappear(path):
    while True:
        playsound.playsound(path)
        global stop_thread_face
        if stop_thread_face:
            break