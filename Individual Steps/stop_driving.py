#stop driving function to run the alarm for 2 loops
def stop_driving(path):
	n=2
	for i in range(0,n):
		playsound.playsound(path)