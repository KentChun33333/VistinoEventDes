import tensorflow as tf

##############################
# Build a Softmax Regression #
##############################

#--------------------------------------------------------------------------
# Step 1 : Make input in tensor Reprsentation  
#--------------------------------------------------------------------------
# 0 = column-dimension, the number on axis=0 equals to the width of data
# 1 = row-dimension, operation along dimension/along axis
#--------------------------------------------------------------------------

x = tf.placeholder(tf.float32, shape=[None, 784]) 
y_ = tf.placeholder(tf.float32, shape=[None, 10])


#--------------------------------------------------------------------------
# Placeholder :
#--------------------------------------------------------------------------
# A value that we'll input when we ask TensorFlow to run a computation.
#
# The input images x will consist of a 2d tensor of floating point numbers. 
# Here we assign it a shape of [None, 784], 
# where 784 is the dimensionality of a single flattened MNIST image, 
# and None indicates that the first dimension, 
# corresponding to the batch size, can be of any size. 
# 
# The target output classes y_ will also consist of a 2d tensor, 
# where each row is a one-hot 10-dimensional vector 
# indicating which digit class the corresponding MNIST image belongs to.
#
# The shape argument to placeholder is optional, 
# but it allows TensorFlow to automatically catch bugs stemming 
# from inconsistent tensor shapes.
#--------------------------------------------------------------------------



W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))

#--------------------------------------------------------------------------
# Variable
#--------------------------------------------------------------------------
# A Variable is a value that lives in TensorFlow's computation graph. 
# It can be used and even modified by the computation. 
# In machine learning applications, 
# one generally has the model parameters be Variables.


#--------------------------------------------------------------------------
# tf.matmul is not element wise
#--------------------------------------------------------------------------
# 2-D tensor [a]
# a = tf.constant([1, 2, 3, 4, 5, 6], shape=[2, 3]) => [[1. 2. 3.]
#                                                       [4. 5. 6.]]
# 2-D tensor [b]
# b = tf.constant([7, 8, 9, 10, 11, 12], shape=[3, 2]) => [[7. 8.]
#                                                          [9. 10.]
#                                                          [11. 12.]]
# c = tf.matmul(a, b) => [[58 64]
#                         [139 154]]


y = tf.nn.softmax(tf.matmul(x, W) + b)
y_ = tf.placeholder(tf.float32, [None, 10])

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))


#--------------------------------------------------------------------------
# tf.reduce_mean(input_tensor, 
#	reduction_indices=None, keep_dims=False, name=None)
#--------------------------------------------------------------------------
#
# Computes the mean of elements across dimensions of a tensor.
# 
# Reduces input_tensor along the dimensions given in reduction_indices. 
# Unless keep_dims is true, 
# the rank of the tensor is reduced by 1 for each entry in reduction_indices. 
# If keep_dims is true, the reduced dimensions are retained with length 1.
# If reduction_indices has no entries, all dimensions are reduced, 
# and a tensor with a single element is returned.

# 'x' is [[1., 1.]
#         [2., 2.]]
# 
# [ By all elements / universe Set ]
# tf.reduce_mean(x) ==> 1.5 
# 
# [ By column-group Set / Colum-Dimension =0 ]
# tf.reduce_mean(x, 0) ==> [1.5, 1.5] 
# 
# [ By row-group Set / Raw-Dimension / =1 ]
# tf.reduce_mean(x, 1) ==> [1.,  2.] 
#--------------------------------------------------------------------------
# Args:
#--------------------------------------------------------------------------
# input_tensor: The tensor to reduce. Should have numeric type.
# reduction_indices: The dimensions to reduce. 
# If None (the default), reduces all dimensions.
# 
# keep_dims: If true, retains reduced dimensions with length 1.
# name: A name for the operation (optional).
# Returns: The reduced tensor.
#--------------------------------------------------------------------------


# tf.reduce_sum(input_tensor, 
#	reduction_indices=None, keep_dims=False, name=None)
#--------------------------------------------------------------------------
# 1. Computes the sum of elements across dimensions of a tensor.
# 2. Reduces input_tensor along the dimensions given in reduction_indices. 
# 3. Unless keep_dims is true, the rank of the tensor is reduced by 1 
#    for each entry in reduction_indices. If keep_dims is true, 
#    the reduced dimensions are retained with length 1.
# 4. If reduction_indices has no entries, all dimensions are reduced, 
#    and a tensor with a single element is returned.
#--------------------------------------------------------------------------
 

#--------------------------------------------------------------------------
# [row]
#--------------------------------------------------------------------------
# x = tf.Variable([1.0, 2.0])
# 1.0
# 2.0
#--------------------------------------------------------------------------
# [column]
#--------------------------------------------------------------------------
# matrix1 = tf.constant([[3., 3.]]) 
# 3.000, 3.000
#--------------------------------------------------------------------------

Training with dataset, note that training is different from detection/recognition

Step 1 input image with label
Image with specification resolution witch is a folder 
Background = 0
X = image 



filename_queue = tf.train.string_input_producer(['/Users/HANEL/Desktop/tf.png']) 
#  list of files to read

reader = tf.WholeFileReader()
key, value = reader.read(filename_queue)

my_img = tf.image.decode_png(value) # use png or jpg decoder based on your files.

init_op = tf.initialize_all_variables()

img_list = []
with tf.Session() as sess:
    sess.run(init_op)
    # Start populating the filename queue.
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)    
    for i in range(1): #length of your filename list
        image = my_img.eval() #here is your image Tensor :) 
    print(image.shape)
    img_list.append(image)

coord.request_stop()
coord.join(threads)






