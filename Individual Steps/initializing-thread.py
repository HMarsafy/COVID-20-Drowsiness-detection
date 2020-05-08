 #intializing a thread and make it start its job.
 t = Thread(target= function, args=(arguments of the function,))
 t.deamon = True
 t.start()