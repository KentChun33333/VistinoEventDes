 # coding: utf-8
import sys
#sys.path.append('/usr/local/lib/python2.7/site-packages')
from six.moves import urllib
import socket
import cv2
import numpy as np
import os

#timeout setting for case redirect or chinese URL 
timeout = 10
socket.setdefaulttimeout(timeout)

# get training data
def store_raw_images():
		# #sport
    # neg_images_link = 'http://image-net.org/api/text/imagenet.synset.geturls?wnid=n00445226'   
    # #car
    # neg_images_link = 'http://image-net.org/api/text/imagenet.synset.geturls?wnid=n04285008'
		# watch
    neg_images_link = 'http://image-net.org/api/text/imagenet.synset.geturls?wnid=n02708433'
    file_path = 'pos'
    img_size = (100, 100)
   
    # neg_image_urls = urllib2.urlopen(neg_images_link).read().decode()
    neg_image_urls = urllib.request.urlopen(neg_images_link).read() 
    pic_num = 1
    
    if not os.path.exists(file_path):
        os.makedirs(file_path)
        
    for i in neg_image_urls.split('\n'):
        try:
            print(i)
            urllib.request.urlretrieve(i, file_path+"/"+str(pic_num)+".jpg")
            img = cv2.imread(file_path+"/"+str(pic_num)+".jpg",cv2.IMREAD_GRAYSCALE)
            # should be larger than samples / pos pic (so we can place our image on it)
            resized_image = cv2.resize(img, img_size)
            cv2.imwrite(file_path+"/"+str(pic_num)+".jpg",resized_image)

            # Check if image not found 
            template = cv2.imread('Uglies/NFound.jpg',0)
            res = cv2.matchTemplate(resized_image,template,cv2.TM_CCOEFF_NORMED)
            threshold = 0.8
            if(res < threshold):
            	print pic_num
            	pic_num += 1
            
        except Exception as e:
        	 print(str(e))

# create text file
def create_pos_n_neg(file):
    for file_type in [file]:
        
        for img in os.listdir(file_type):

            if file_type == 'pos':
                line = file_type+'/'+img+' 1 0 0 100 100\n'
                with open('info.dat','a') as f:
                    f.write(line)
            elif file_type == 'neg':
                line = file_type+'/'+img+'\n'
                with open('bg.txt','a') as f:
                    f.write(line)

#store_raw_images()
create_pos_n_neg('pos')