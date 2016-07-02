
import cv2 
import numpy as np
from image_process_func import pyramid, sliding_window, non_max_suppression

class ObjectDetrector:
	def __init__(self, model, desc):
		self.model = model 
		self.desc = desc



DEF : 
- desciptor containor 
input : sliding_window_object 
output : label_data with confidence
    - Binary Case 
    - Multi-case 

