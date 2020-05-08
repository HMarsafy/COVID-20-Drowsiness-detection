#function to start the alarm when the the driver close his eyes for a while , and to stop it when the flag changeg
def alarm_sound(path):
    while True:
        playsound.playsound(path)
        global stop_thread
        if stop_thread:
            break