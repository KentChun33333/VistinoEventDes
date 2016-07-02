

import numpy as np
import argparse
import imutils
import cv2
import matplotlib.pyplot as plt

def get_args():
    '''This function parses and return arguments passed in'''
    # Assign description to the help doc
    parser = argparse.ArgumentParser(
        description=''' Script read both depth and rgb frames from kinetic 
        and save them in Assigned Folder''')
    # Add arguments
    parser.add_argument(
        '-fn', '--folder_name', type=str, help='file name.txt', required=True)
    # Array for all arguments passed to script
    args = parser.parse_args()
    # Assign args to variables
    folder_name = args.folder_name
    # Return all variable values
    return folder_name



with open(the_filename, 'r') as f:
    my_list = [line.rstrip('\n') for line in f]



def data_clean(inTxT):
	repls = {'[' : '', ']' : '','(':'',')':''}
	tmp = reduce(lambda a, kv: a.replace(*kv), repls.iteritems(), inTxT)
	frame_id, x , y = tmp.split(',')
	return int(frame_id), int(x) , int(y)

# trim some error
my_list=my_list[3:len(my_list)-3]

# List Comprehension
tmp = [ data_clean(s) for s in my_list ]

X = []
Y = []
points = []
for frame_id, x , y in tmp:
	X.append(int(x))
	Y.append(int(y))
	points.append((int(x),int(y)))

plt.plot(X, Y, 'r')
plt.show()

plt.plot(X, Y, 'ro')
plt.show()

# use VQ to compress the data into fix-lenth, then we could traning the data 
import vq_script
strokes = vq_script.getGesture(points)
#segments = vq_script.getSegments(points)
#strokeText = vq_script.getGestureStr(strokes)

# Histogram for strokes
from collections import Counter
cnt = Counter(strokes)
# normal-lize the feature 
norm1 = x / np.linalg.norm(x)

# plot Histogram 
plt.plot(cnt.keys(), cnt.values(), 'r--',linewidth=3)
plt.bar(cnt.keys(), cnt.values(), 0.6, color="blue")
plt.show()


#####################
# Combined Versiton #
#####################
import numpy as np
import argparse
import imutils
import cv2
import matplotlib.pyplot as plt
import vq_script
# Data read 
the_filename_list = ['circle_wave.txt','horizontal_wave.txt', 
'virtical_wave.txt']
def data_clean(inTxT):
    repls = {'[' : '', ']' : '','(':'',')':''}
    tmp = reduce(lambda a, kv: a.replace(*kv), repls.iteritems(), inTxT)
    frame_id, x , y = tmp.split(',')
    return int(frame_id), int(x) , int(y)

def fast_vq_file(the_filename):
    with open(the_filename, 'r') as f:
        my_list = [line.rstrip('\n') for line in f]
    # trim some error
    my_list=my_list[3:len(my_list)-3]    
    # List Comprehension
    tmp = [ data_clean(s) for s in my_list ]    
    X = []
    Y = []
    points = []
    for frame_id, x , y in tmp:
        X.append(int(x))
        Y.append(int(y))
        points.append((int(x),int(y)))
    return points

#
def fast_train_clean(points, label_index):
    import vq_script
    new_X_train = []
    new_Y_train = []
    for i in range(30):
        points_tmp = points[(i*3):]
        strokes = vq_script.getGesture(points_tmp)
        cnt_tmp = np.histogram([strokes], bins=[1, 2, 3,4,5,6,7,8,9,10])
        # [*]
        new_X_train.append(cnt_tmp[0])
        new_Y_train.append([label_index])    

        points_tmp = points[:-(i*3+1)]
        # +1 because of the [:-0] whould return [null] 
        strokes = vq_script.getGesture(points_tmp)

        #cnt_tmp = Counter(strokes)
        cnt_tmp = np.histogram([strokes], bins=[1, 2, 3,4,5,6,7,8,9,10])
        # [*]
        new_X_train.append(cnt_tmp[0])
        new_Y_train.append([label_index])
    return new_X_train, new_Y_train

# Main Script - trian  : )

new_X_train = []
new_Y_train = []
for lable, file_name in enumerate(the_filename_list):
    points = fast_vq_file(file_name)
    X, Y   = fast_train_clean(points,lable)
    if len(new_X_train) is 0 :
        new_X_train=X
        new_Y_train=Y   
    else:
        new_X_train = np.concatenate((new_X_train, X), axis=0)
        new_Y_train = np.concatenate((new_Y_train, Y), axis=0)


# Main Script -> test
def main_testX(file_name):
    points = fast_vq_file(file_name)
    X, Y   = fast_train_clean(points,-1)
    testX = np.array([x / np.linalg.norm(x) for x in X ])
    # in the interactive sess, 
    return testX

def test_visualisation(x):
    # plot Histogram 
    plt.plot([1,2,3,4,5,6,7,8,9], x, 'r--',linewidth=3)
    plt.bar([1,2,3,4,5,6,7,8,9], x, 0.6, color="blue")
    plt.show()

def remove_stationay_five(in2D):
    for i in in2D:
        i[4] =0
    return in2D

testX = main_testX('testing.txt')
testX2 = main_testX('testing2.txt')

######
# Softmax (TF)
######
import tensorflow as tf 

# Parameters
learning_rate = 0.01
training_epochs = 25
batch_size = 30 # must could be divided by train_ex_numbers
display_step = 1
##======

# trX, trY, teX, teY = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels
new_X_train_nor  = np.array([x / np.linalg.norm(x) for x in new_X_train ])
# Be Aware of the np.array list comprehension should be like above
trX = new_X_train_nor
trY_tmp = []
for i in new_Y_train:
    # convert to one hot vector
    tmp = np.histogram([i], bins=[0,1, 2, 3])
    trY_tmp.append(tmp[0])
trY = np.array(trY_tmp)


trX = trX.astype('float32')
trY = trY.astype('float32')
#    if len(trY) == 0:
#        trY = tmp[0]
#    else:
#        trY = np.concatenate((trY, tmp[0]), axis=1)

train_ex_numbers, inputDim = trX.shape
_, labelNum = trY.shape

##======
x = tf.placeholder(tf.float32, [None, inputDim])
y = tf.placeholder(tf.float32, [None, labelNum]) 

# Set model weights
W = tf.Variable(tf.zeros([inputDim, labelNum]))
b = tf.Variable(tf.zeros([labelNum]))

# Construct model
activation = tf.nn.softmax(tf.matmul(x, W) + b) # Softmax
###y = tf.nn.softmax(tf.matmul(x,W) + b)
# Construct model
activation = tf.nn.softmax(tf.matmul(x, W) + b) # Softmax

# Minimize error using cross entropy
cost = tf.reduce_mean(-tf.reduce_sum(y*tf.log(activation), reduction_indices=1)) 
# Cross entropy
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost) 

# Initializing the variables
init = tf.initialize_all_variables()

sess = tf.InteractiveSession()
sess.run(init)
####
#
###
with tf.Session() as sess:
    sess.run(init)
    for epoch in range(training_epochs*100):
        avg_cost = 0.
        #--------------------------------------------------------
        # total_batch = int(mnist.train.num_examples/batch_size)
        #--------------------------------------------------------
        total_batch = int(train_ex_numbers)/batch_size # 6
        for start, end in zip(range(0, len(trX), 30), range(30, len(trX), 30)):
            avg_cost = sess.run(optimizer, feed_dict={x: trX[start:end], y: trY[start:end]})
            print str(avg_cost)
            # [*] start , end  = (0   ,30)
            # [*] start , end  = (30  ,60)
            # [*] start , end  = (60  ,90)
            # [*] start , end  = (90  ,120)
            # [*] start , end  = (120 ,150)
            # [*] Notice the feed_dict must corresponding to the top of variables
        #  avg_cost += sess.run(optimizer, feed_dict={x: trX[start:end], y: trY[start:end]})/5
        # Display logs per epoch step
        #####if epoch % display_step == 0:
            #####print "Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost)
    print "Optimization Finished!"
    # Test model 1
    correct_prediction = tf.equal(tf.argmax(activation, 1), tf.argmax(y, 1))
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print "Accuracy:"
    print accuracy.eval({x: trX, y: trY})
    #print accuracy.eval({x: testX[start+20:end+20], y: trY[start+20:end+20]})


    # Test Modle 2
    # Show the %%%%
    print activation.eval(feed_dict={x: trX[165:170]},session=sess)
    # Test Modle 3
    prediction=tf.argmax(activation,1)
    print prediction.eval(feed_dict={x: trX[160:162]},session=sess)

# sess2 train with remove_stationay_five()
print activation.eval(feed_dict={x: trX[165:170]},session=sess)
print prediction.eval(feed_dict={x: trX[160:162]},session=sess)
print activation.eval(feed_dict={x: trX[165:170]},session=sess2)
print prediction.eval(feed_dict={x: trX[160:162]},session=sess2)




print activation.eval(feed_dict={x: testX},session=sess)
print prediction.eval(feed_dict={x: testX},session=sess)
print activation.eval(feed_dict={x: testX_2},session=sess2)
print prediction.eval(feed_dict={x: testX_2},session=sess2)
#   print "Accuracy:", accuracy.eval({x: mnist.test.images, y: mnist.test.labels})




def save(checkpoint_file=’hello.chk’):
    with tf.Session() as session:
        x = tf.Variable([42.0, 42.1, 42.3], name='x')
        y = tf.Variable([[1.0, 2.0], [3.0, 4.0]], name='y')
        not_saved = tf.Variable([-1, -2], name='not_saved')
        session.run(tf.initialize_all_variables())

        print(session.run(tf.all_variables()))
        saver = tf.train.Saver([x, y])
        saver.save(session, checkpoint_file)

def restore(checkpoint_file=’hello.chk’):
    x = tf.Variable(-1.0, validate_shape=False, name='x')
    y = tf.Variable(-1.0, validate_shape=False, name='y')
    with tf.Session() as session:
        saver = tf.train.Saver()
        saver.restore(session, checkpoint_file)
        print(session.run(tf.all_variables()))

def reset():
    tf.reset_default_graph()
