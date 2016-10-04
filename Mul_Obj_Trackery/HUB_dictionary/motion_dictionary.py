################################
# [*] Motion Interpretation    #
################################
#   up-left    up   up-right   #
#          7   8   9           #
#     left 4   5   6 right     #
#          1   2   3           #
# down-left   down  down-right #
################################


class Motion_Dictionary():
	def __init__(self):
		self.motionDictionary ={}

		self.motionDictionary['Moving Down-Left' ]  = 1
		self.motionDictionary['Moving Down'      ]  = 2
		self.motionDictionary['Moving Down-Right']  = 3
		self.motionDictionary['Moving Left'      ]  = 4
		self.motionDictionary['Stationary'       ]  = 5
		self.motionDictionary['Moving Right'     ]  = 6
		self.motionDictionary['Moving Up-Left'   ]  = 7
		self.motionDictionary['Moving Up'        ]  = 8
		self.motionDictionary['Moving Up-Right'  ]  = 9

	def search(self,value):
		for a,b in self.motionDictionary.iteritems():
			if b == int(value):
				return a


class Motion_3D_Dictionary():
	def __init__(self):
		self.motionDictionary ={}

		# Z = +1
		self.motionDictionary['DOWNLEFT' ]  = 1
		self.motionDictionary['DOWN'     ]  = 2
		self.motionDictionary['DOWNRIGHT']  = 3
		self.motionDictionary['LEFT'     ]  = 4
		self.motionDictionary['Moving Up'   ]  = 5
		self.motionDictionary['RIGHT'    ]  = 6
		self.motionDictionary['UPLEFT'   ]  = 7
		self.motionDictionary['UP'       ]  = 8
		self.motionDictionary['UPRIGHT'  ]  = 9
		# Z = 0
		self.motionDictionary['Moving Down-Left' ]  = 1+9
		self.motionDictionary['Moving Down'      ]  = 2+9
		self.motionDictionary['Moving Down-Right']  = 3+9
		self.motionDictionary['Moving Left'      ]  = 4+9
		self.motionDictionary['Stationary'       ]  = 5+9
		self.motionDictionary['Moving Right'     ]  = 6+9
		self.motionDictionary['Moving Up-Left'   ]  = 7+9
		self.motionDictionary['Moving Up'        ]  = 8+9
		self.motionDictionary['Moving Up-Right'  ]  = 9+9
		# Z = -1
		self.motionDictionary['DOWNLEFT' ]  = 1+18
		self.motionDictionary['DOWN'     ]  = 2+18
		self.motionDictionary['DOWNRIGHT']  = 3+18
		self.motionDictionary['LEFT'     ]  = 4+18
		self.motionDictionary['Moving Down'   ]  = 5+18
		self.motionDictionary['RIGHT'    ]  = 6+18
		self.motionDictionary['UPLEFT'   ]  = 7+18
		self.motionDictionary['UP'       ]  = 8+18
		self.motionDictionary['UPRIGHT'  ]  = 9+18

	def check(self,value):
		for a,b in self.motionDictionary.iteritems():
			if b == int(value):
				return a

