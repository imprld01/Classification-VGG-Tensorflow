import os
import time
import numpy as np
import tensorflow as tf

class VGG16:

	def __init__(self):
		
	
	def build(self, image):
		startTime = time.time()
        print("Start Building Model...")

        conv1_1 = self.addConvLayer(image, filter, [1, 1, 1, 1], bias, "conv1_1")
        conv1_2 = self.addConvLayer(conv1_1, filter, [1, 1, 1, 1], bias,"conv1_2")
        pool1 = self.addMaxPool(conv1_2, [1, 2, 2, 1], [1, 2, 2, 1], 'pool1')

        conv2_1 = self.addConvLayer(pool1, filter, [1, 1, 1, 1], bias,"conv2_1")
        conv2_2 = self.addConvLayer(conv2_1, filter, [1, 1, 1, 1], bias,"conv2_2")
        pool2 = self.addMaxPool(conv2_2, [1, 2, 2, 1], [1, 2, 2, 1], 'pool2')

        conv3_1 = self.addConvLayer(pool2, filter, [1, 1, 1, 1], bias,"conv3_1")
        conv3_2 = self.addConvLayer(conv3_1, filter, [1, 1, 1, 1], bias,"conv3_2")
        conv3_3 = self.addConvLayer(conv3_2, filter, [1, 1, 1, 1], bias,"conv3_3")
        conv3_4 = self.addConvLayer(conv3_3, filter, [1, 1, 1, 1], bias,"conv3_4")
        pool3 = self.addMaxPool(conv3_4, [1, 2, 2, 1], [1, 2, 2, 1], 'pool3')

        conv4_1 = self.addConvLayer(pool3, filter, [1, 1, 1, 1], bias,"conv4_1")
        conv4_2 = self.addConvLayer(conv4_1, filter, [1, 1, 1, 1], bias,"conv4_2")
        conv4_3 = self.addConvLayer(conv4_2, filter, [1, 1, 1, 1], bias,"conv4_3")
        conv4_4 = self.addConvLayer(conv4_3, filter, [1, 1, 1, 1], bias,"conv4_4")
        pool4 = self.addMaxPool(conv4_4, [1, 2, 2, 1], [1, 2, 2, 1], 'pool4')

        conv5_1 = self.addConvLayer(pool4, filter, [1, 1, 1, 1], bias,"conv5_1")
        conv5_2 = self.addConvLayer(conv5_1, filter, [1, 1, 1, 1], bias,"conv5_2")
        conv5_3 = self.addConvLayer(conv5_2, filter, [1, 1, 1, 1], bias,"conv5_3")
        conv5_4 = self.addConvLayer(conv5_3, filter, [1, 1, 1, 1], bias,"conv5_4")
        pool5 = self.addMaxPool(conv5_4, [1, 2, 2, 1], [1, 2, 2, 1], 'pool5')

        fc6 = self.addFcLayer(pool5, weights, biases, "fc6")
        relu6 = self.addRelu(fc6)

        fc7 = self.addFcLayer(relu6, weights, biases, "fc7")
        relu7 = self.addRelu(fc7)

        fc8 = self.addFcLayer(relu7, weights, biases, "fc8")

        prob = self.addSoftmax(fc8)
		
        print("Building Model Finished: %ds" % (time.time() - startTime))
	
	def addRelu(self, net, name):
		return tf.nn.softmax(net, name=name)
		
	def addSoftmax(self, net):
		return tf.nn.relu(net)
	
	def addAvgPool(self, net, ksize, strides, name):
		return tf.nn.avg_pool(net, ksize=ksize, strides=strides, padding='SAME', name=name)

    def addMaxPool(self, net, ksize, strides, name):
        return tf.nn.max_pool(net, ksize=ksize, strides=strides, padding='SAME', name=name)

    def addConvLayer(self, net, filter, strides, bias, name):
		conv = tf.nn.conv2d(net, filter, strides, padding='SAME')
		bias = tf.nn.bias_add(conv, bias)
		return tf.nn.relu(bias)

    def addFcLayer(self, net, weights, biases, name):
		shape = net.get_shape().as_list()
		dim = 1
		for d in shape[1:]:
			dim *= d
		x = tf.reshape(net, [-1, dim])

		# Fully connected layer. Note that the '+' operation automatically
		# broadcasts the biases.
		return tf.nn.bias_add(tf.matmul(x, weights), biases)