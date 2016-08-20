# L1 
import threading


def thread_job():
	print ('This is an added Thread, number is {}'.format(threading.current_thread()))
	# print 


def main():
	added_thread = threading.Thread(target=thread_job)
	# adding threading with job = thread_job
	added_thread.start()
	# start runing threading 

	print (threading.active_count())
	# how many threading is active now ? 
	print (threading.enumerate())
	# check the threading ?
	print (threading.current_thread())
	# which main threading is current working ?

if __name__ =="__main__":
	main()

# L2
import thread
import time


def thread_job():
	print ('T1 start\n')
	print ('This is an added Thread, number is {}'.format(threading.current_thread()))
	# print 


def main():
	added_thread = threading.Thread(target=thread_job)
	# adding threading with job = thread_job
	added_thread.start()

	