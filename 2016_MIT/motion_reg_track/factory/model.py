#########
# Author : Kent Chiu 
# Note   : Most of simple model could be build by sklearn 
# EX     : from sklearn.svm import SVC 
#          from sklearn.tree import DecisionTreeClassifier
#
import tensorfow as tf
import numpy as np


def img_lable_generator(posFolds):
    '''posFole is an object'''
    init = 1
    result = []
    for singleFoler in posFolds:
        for i in singleFoler:
            
        tmp = [0]*(len(posFolds)+1) # add 1 as background]
        tmp[init] = 1
        result.append(tmp)
        init+=1






class Softmax:
    def __init__(self, learning_rate=0.01, training_epochs=100, ):
        # Set container parameter
        # Parameters
        sel.learning_rate = learning_rate
        self.training_epochs = training_epochs

        learning_rate = 0.01
        # batch_size = 100 this is not need to be ....
        display_step = 1

    def training(self, X,Y):
        '''where X is the flatten image'''

    def fit(self, X, Y):
        # note the Labels must use list-comprehension 

        return
    def detect(self):
        # retrive the model_conf and predict

        return

def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))


def model(X, w):
    return tf.matmul(X, w) 
    # notice we use the same model as linear regression, this is because there is a baked in cost function which performs softmax and cross entropy


mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
trX, trY, teX, teY = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels

X = tf.placeholder("float", [None, 784]) # create symbolic variables
Y = tf.placeholder("float", [None, 10])

w = init_weights([784, 10]) # like in linear regression, we need a shared variable weight matrix for logistic regression

py_x = model(X, w)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(py_x, Y)) # compute mean cross entropy (softmax is applied internally)
train_op = tf.train.GradientDescentOptimizer(0.05).minimize(cost) # construct optimizer
predict_op = tf.argmax(py_x, 1) # at predict time, evaluate the argmax of the logistic regression



####
#
####

import tensorflow as tf
# where inputDim is the number of dimension of a single tensor ?

##======

trX, trY, teX, teY = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels
_, inputDim = trX.shape
_, labelNum = trY.shape

##======
x = tf.placeholder(tf.float32, [None, inputDim])
W = tf.Variable(tf.zeros([inputDim, labelNum]))
b = tf.Variable(tf.zeros([labelNum]))

y = tf.nn.softmax
y_ = tf.placeholder(tf.float32, [None, labelNum])

cross_entropy = -tf.reduce_sum(y_*tf.log(y))
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)


for i in range(1000):
  batch_xs, batch_ys = mnist.train.next_batch(100)
  sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))



#####
# theory implementation
####


f = np.array([123, 456, 789]) # example with 3 classes and each having large scores
p = np.exp(f) / np.sum(np.exp(f)) # Bad: Numeric problem, potential blowup

# instead: first shift the values of f so that the highest number is 0:
f -= np.max(f) # f becomes [-666, -333, 0]
p = np.exp(f) / np.sum(np.exp(f)) # safe to do, gives the correct answer