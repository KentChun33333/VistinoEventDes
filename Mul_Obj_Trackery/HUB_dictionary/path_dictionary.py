__author__ = '''Kent (Jin-Chun Chiu)'''

import numpy as np
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw

##################################
# [*] Path Interpretation        #
##################################
#   up-left     up    up-right   #
#          7:14    8    9:15     #
#     left 4    5    6:0 right   #
#          1:11   2:10  3:13     #
# down-left    down   down-right #
##################################

# Notice that if not interactive mode,
# the image is Up-Down reversed


class Path_DTW_Dictionary_InterActive():
	def __init__(self):
		self.pathDictionary ={}
		# you could pre-difined some pattern there

		self.pathDictionary['Slow Moving Down-Left'   ]  = [1]*8
		self.pathDictionary['Slow Moving Down'        ]  = [2]*8
		self.pathDictionary['Slow Moving Down-Right'  ]  = [3]*8
		self.pathDictionary['Slow Moving Left'        ]  = [4]*8
		self.pathDictionary['Keeping Stationary'      ]  = [5]*8
		self.pathDictionary['Slow Moving Right'       ]  = [6]*8
		self.pathDictionary['Slow Moving Up-Left'     ]  = [7]*8
		self.pathDictionary['Slow Moving Up'          ]  = [8]*8
		self.pathDictionary['Slow Moving Up-Right'    ]  = [9]*8

#		self.pathDictionary['Clockwise Circle'        ]  = [8,15,6,13,10,11,4,14]
#		self.pathDictionary['Clockwise Circle.'       ]  = [15,6,13,10,11,4,14,8]
#		self.pathDictionary['Clockwise Circle..'      ]  = [6,13,10,11,4,14,8,15]
#		self.pathDictionary['Clockwise Circle...'     ]  = [13,10,11,4,14,8,15,6]
#		self.pathDictionary['Clockwise Circle....'    ]  = [10,11,4,14,8,15,6,13]
#		self.pathDictionary['Clockwise Circle....'    ]  = [11,4,14,8,15,6,13,10]
#		self.pathDictionary['Clockwise Circle.....'   ]  = [4,14,8,15,6,13,10,11]
#		self.pathDictionary['Clockwise Circle......'  ]  = [14,8,15,6,13,10,11,4]
#		self.pathDictionary['Conter Clockwise Circle' ]  = [14,4,11,10,13,6,15,8]
#		self.pathDictionary['Conter Clockwise Circle.' ]  = [4,11,10,13,6,15,8,14]
#		self.pathDictionary['Conter Clockwise Circle..' ]  = [11,10,13,6,15,8,14,4]
#		self.pathDictionary['Conter Clockwise Circle...' ]  = [10,13,6,15,8,14,4,11]
#		self.pathDictionary['Conter Clockwise Circle....' ]  = [13,6,15,8,14,4,11,10]
#		self.pathDictionary['Conter Clockwise Circle.....' ]  = [6,15,8,14,4,11,10,13]
#		self.pathDictionary['Conter Clockwise Circle......' ]  = [15,8,14,4,11,10,13,6]
#		self.pathDictionary['Conter Clockwise Circle.......' ]  = [8,14,4,11,10,13,6,15]#

#		self.pathDictionary['Horizontal Wave'         ]  = [4,0,4,0,4,0,4,0]
#		self.pathDictionary['Vertical Wave'           ]  = [10,8,10,8,10,8,10,8]#

#		self.pathDictionary['Horizontal Wave'         ]  = [0,4,0,4,0,4,0,4]
#		self.pathDictionary['Vertical Wave'           ]  = [8,10,8,10,8,10,8,10]


	def search(self,value):
		# type(value) = list
		memo = []
		memo_distance = []

		for pathStr, pathList in self.pathDictionary.iteritems():
			distance, path = fastdtw(value, pathList, dist=euclidean)
			memo.append(pathStr)
			memo_distance.append(distance)
		# print ('[*] Min_DTW_Distance : %s'%(str(min(memo_distance))) )
		if min(memo_distance) < 40:
			idx = memo_distance.index(min(memo_distance))
			return memo[idx]
		else:
			return 'None'

	def search2(self,value):
		# type(value) = list
		memo = []
		memo_distance = []

		new_value = [] # spare 5 #
		for tmp in value:
			if tmp == 2 :
				new_value.append(10)
			else :
				new_value.append(tmp)


		for pathStr, pathList in self.pathDictionary.iteritems():
			distance, path = fastdtw(new_value, pathList, dist=euclidean)
			memo.append(pathStr)
			memo_distance.append(distance)
		print ('[*] Min_DTW_Distance : %s'%(str(min(memo_distance))) )
		if min(memo_distance) < 40:
			idx = memo_distance.index(min(memo_distance))
			return memo[idx]
		else:
			return 'None'


class Path_DTW_Dictionary():
    def __init__(self):
        self.pathDictionary ={}
        # you could pre-difined some pattern there

        self.pathDictionary['Moving UP-Left'     ]  = [1]*8
        self.pathDictionary['Moving UP'          ]  = [2]*8
        self.pathDictionary['Moving UP-Right'    ]  = [3]*8
        self.pathDictionary['Slow Moving Left'   ]  = [4]*8
        self.pathDictionary['Keeping Stationary' ]  = [5]*8
        self.pathDictionary['Moving Right'       ]  = [6]*8
        self.pathDictionary['Moving Down-Left'   ]  = [7]*8
        self.pathDictionary['Moving Down'        ]  = [8]*8
        self.pathDictionary['Moving Down-Right'  ]  = [9]*8

#       self.pathDictionary['Clockwise Circle'        ]  = [8,15,6,13,10,11,4,14]
#       self.pathDictionary['Clockwise Circle.'       ]  = [15,6,13,10,11,4,14,8]
#       self.pathDictionary['Clockwise Circle..'      ]  = [6,13,10,11,4,14,8,15]
#       self.pathDictionary['Clockwise Circle...'     ]  = [13,10,11,4,14,8,15,6]
#       self.pathDictionary['Clockwise Circle....'    ]  = [10,11,4,14,8,15,6,13]
#       self.pathDictionary['Clockwise Circle....'    ]  = [11,4,14,8,15,6,13,10]
#       self.pathDictionary['Clockwise Circle.....'   ]  = [4,14,8,15,6,13,10,11]
#       self.pathDictionary['Clockwise Circle......'  ]  = [14,8,15,6,13,10,11,4]
#       self.pathDictionary['Conter Clockwise Circle' ]  = [14,4,11,10,13,6,15,8]
#       self.pathDictionary['Conter Clockwise Circle.' ]  = [4,11,10,13,6,15,8,14]
#       self.pathDictionary['Conter Clockwise Circle..' ]  = [11,10,13,6,15,8,14,4]
#       self.pathDictionary['Conter Clockwise Circle...' ]  = [10,13,6,15,8,14,4,11]
#       self.pathDictionary['Conter Clockwise Circle....' ]  = [13,6,15,8,14,4,11,10]
#       self.pathDictionary['Conter Clockwise Circle.....' ]  = [6,15,8,14,4,11,10,13]
#       self.pathDictionary['Conter Clockwise Circle......' ]  = [15,8,14,4,11,10,13,6]
#       self.pathDictionary['Conter Clockwise Circle.......' ]  = [8,14,4,11,10,13,6,15]#

#       self.pathDictionary['Horizontal Wave'         ]  = [4,0,4,0,4,0,4,0]
#       self.pathDictionary['Vertical Wave'           ]  = [10,8,10,8,10,8,10,8]#

#       self.pathDictionary['Horizontal Wave'         ]  = [0,4,0,4,0,4,0,4]
#       self.pathDictionary['Vertical Wave'           ]  = [8,10,8,10,8,10,8,10]


    def search(self,value):
        # type(value) = list
        memo = []
        memo_distance = []

        for pathStr, pathList in self.pathDictionary.iteritems():
            distance, path = fastdtw(value, pathList, dist=euclidean)
            memo.append(pathStr)
            memo_distance.append(distance)
        # print ('[*] Min_DTW_Distance : %s'%(str(min(memo_distance))) )
        if min(memo_distance) < 40:
            idx = memo_distance.index(min(memo_distance))
            return memo[idx]
        else:
            return 'None'

    def search2(self,value):
        # type(value) = list
        memo = []
        memo_distance = []

        new_value = [] # spare 5 #
        for tmp in value:
            if tmp == 2 :
                new_value.append(10)
            else :
                new_value.append(tmp)


        for pathStr, pathList in self.pathDictionary.iteritems():
            distance, path = fastdtw(new_value, pathList, dist=euclidean)
            memo.append(pathStr)
            memo_distance.append(distance)
        print ('[*] Min_DTW_Distance : %s'%(str(min(memo_distance))) )
        if min(memo_distance) < 40:
            idx = memo_distance.index(min(memo_distance))
            return memo[idx]
        else:
            return 'None'
