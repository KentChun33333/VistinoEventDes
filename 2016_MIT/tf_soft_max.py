import tensorflow as tf



from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

x = tf.placeholder(tf.float32, [None, 784])

###############################################
# tf.placeholder ( dtype, shape, name )
#
###

import tensorflow as tf

W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))


y = tf.nn.softmax(tf.matmul(x, W) + b)
'''
'''

# this is the place to put the correct answer of after the crossing_entropy 
y_ = tf.placeholder(tf.float32, [None, 10])

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

#
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)


init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)



for i in range(1000):
  batch_xs, batch_ys = mnist.train.next_batch(100)
  sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

# Save the variables to disk.

saver = tf.train.Saver()  # defaults to saving all variables - in this case w and b
save_path = saver.save(sess, "model.ckpt")
print("Model saved in file: %s" % save_path)

########################################
# Testing Model                       
#---------------------------------------
# Note : The sess itself is the model  
#################

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))


###############################
# Restore Variables from Disk #
###############################

# Create same variables. and do not neet to init
v1 = tf.Variable(..., name="v1")
v2 = tf.Variable(..., name="v2")


# Add ops to save and restore all the variables.
saver = tf.train.Saver()

# Later, launch the model, use the saver to restore variables from disk, and
# do some work with the model.
with tf.Session() as sess:
  # Restore variables from disk.
  saver.restore(sess, "/tmp/model.ckpt")
  print("Model restored.")
  # Do some work with the model
  ...

