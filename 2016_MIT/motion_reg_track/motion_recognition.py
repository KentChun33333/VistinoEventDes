
#------------------------------------------|
# Object  Detection                        |
# Object  Recognition  (Rigid-Object)      |
# Gesture Detection                        |
# Gesture Recognition  (Non-Rigid-Object)  |
# Motion  Detection                        |
# Motion  Recognition                      |
#

# A Trajectory-based Approach to Gesture Recognition
# 動態手勢辨識
# 手語辨識
# 自我組織特徵映射圖網路
# 適應共振理論
# adaptive resonance theory (ART)
# self-organizing feature maps (SOM)
# dynamic gesture recognition;character recognition
# sign language recognition

# 手勢辨識系統可廣泛應用於人機介面設計、醫療復健、虛擬實境、
# 數位藝術創作與遊戲設計等領域，尤其是手語辨識系統，
# 更是需要搭配準確且可行的手勢辨識系統。 
# 本論文提出SOMART演算法，將動態手勢辨識之問題轉換為軌跡辨識問題處理。
# SOMART演算法主要包含兩個步驟，首先，
# 將多維的手勢資訊利用SOM網路作基本手形分類器並投影至二維平面中。
# 接著，將前一步驟所產生的平面軌跡輸入至改良後的ART網路做圖樣的辨識以辨識動態手勢。
# 另外，我們利用「以軌跡辨識為基礎」的辨識概念，進行手部移動軌跡辨識，
# 同樣可解決動態時序資料辨識的問題。 
# 結果驗證部份，我們定義47種靜態手勢、103種動態手勢及八種手部移動軌跡，
# 分別請十位使用者錄製手部移動軌跡資料，整體資料庫的數量為4650筆資料。
# 靜態手勢的平均辨識率為92%，動態手勢的平均辨識率為88%。而手部移動軌跡的平均辨識率達99%。
# 

# HMM
# Kalman Filter 
# DTW (Dynamic Time Wrapping)
# VQ (Vector Qantization)
# HMM traning 
# Viterbi traning 
# 




Use Deep Learnining to learn the better representation of 
Time-series data

For control : 
if a line , 
if a big circle, 

motion_1 : 
motion_2 : 
motion_3 :
motion_4 : 
motion_5 : 
motion_6 : 

Motion
- motion-detection mode
- 2D motion_pattern_recognition ( repeatable-motion)

# 2D motion 

position_extraction 
- representation 





import numpy as np

### 
def nor(inputArray):
	if type(inputArray)==list:
		inputArray = np.array(inputArray)
	norm = [float(i)/max(inputArray) for i in inputArray]
	return norm

def mean_central(inputArray):
	if type(inputArray)==list:
		inputArray = np.array(inputArray)
	result = [x - inputArray.mean() for x in inputArray]
	return result

# trajactory data
circle
def generate_ran_pair(numMax):
	x = np.random.rand(1,numMax)
	y = np.random.rand(1,numMax)
	return x[0],y[0]

def norm_mean_center(x):
	return mean_central(nor(x))

def norm_mean_center(x,y):
	maxNum = max(max(x),max(y))


x,y  = generate_ran_pair(15)
x,y  = generate_ran_pair(8)

x = [101,102,103,104,105]
y = [1,2,3,4,5]


[]

# x_nor = nor(x)
# y_nor = nor(y)
# x_m_nor = mean_central(x_nor)
# y_m_nor = mean_central(y_nor)


x.std()/ x.mean()
Out[132]: 0.013730228760903837 


y.std()/ y.mean()
Out[133]: 0.47140452079103173


# visulization 
plt.plot(x,y,'or')
# plt.plot(x,y,'s')
plt.plot(norm_mean_center(x),norm_mean_center(y),'or')


#  Test

line 
x = [1,2,3,4,5,6,7,8,9,10]
y = [1,2,3,4,5,6,7,8,9,10]

x = [1,2,3,4,5,6,7,8,9,10]
y = [1,1,1,1,1,1,1,1,1,1]

y = [1,2,3,4,5,6,7,8,9,10]
x = [1,1,1,1,1,1,1,1,1,1]

x = [1,2,3,4,5,6,7,8,9,10]
y = [1,2,3,4,5,6,7,8,9,10]


# motion algo

hand_position = [[1,1],[2,2,],[0,0]]



class SimpleTrack():
	def __init__(self):
		self.positionSet =[]
		
	def track(self,image):










