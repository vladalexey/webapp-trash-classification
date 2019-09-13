import numpy as np 
import tensorflow as tf 

def conv_pool2d(x, filter_size, input_channel, num_filter):

    with tf.name_scope("conv_pool_{}".format(filter_size)):

        filter_shape = [filter_size, filter_size, input_channel, num_filter]
        w = tf.Variable(tf.truncated_normal(filter_shape))
        b = tf.Variable(tf.constant(0.1, shape=[num_filter]))

        layer = tf.nn.conv2d(
            x, 
            w, 
            strides=[1, 1, 1, 1], 
            padding="SAME",
            name="conv")

        layer = tf.nn.bias_add(layer, b)

        h = tf.nn.relu(layer, name='relu')

        pooled = maxp2d(h)

        return pooled

def maxp2d(x, k=2):

    layer = tf.nn.max_pool(
        x,
        ksize=[1, k, k, 1],
        strides=[1, k, k, 1], 
        padding="SAME")

    return layer

class Trash_CNN:
    
    def __init__(self, num_classes, input_shape, filters, input_channel):
        
        self.num_classes = num_classes
        H, W, C = input_shape

        self.input_x = tf.placeholder(tf.float32, shape=[None, H, W, C ], name="input_x")
        self.input_y = tf.placeholder(tf.float32, shape=[None, num_classes], name="input_y")

        layer = conv_pool2d(self.input_x, filters[0], C, input_channel)
        print("Conv_pool2d {}".format(layer.get_shape())) 

        for idx in range(1,len(filters[1:]) + 1):
            
            # num_filter = input_channel * 2 if input_channel % idx == 0 else input_channel
            num_filter = input_channel * 2 
            layer = conv_pool2d(layer, filters[idx], input_channel, num_filter)
            print("Conv_pool2d {}".format(layer.get_shape()))    
            input_channel = num_filter

        shape = int(np.prod(layer.get_shape()[1:]))
        flat = tf.reshape(layer, [-1, shape])

        w = tf.Variable(tf.truncated_normal((shape, 256)))
        b = tf.Variable(tf.constant(0.1, shape=[256]))
 
        fc = tf.add(tf.matmul(flat, w), b)

        print("Dense {}".format(fc.get_shape()))
        fc = tf.nn.relu(fc)

        w = tf.Variable(tf.truncated_normal((256, num_classes)))
        b = tf.Variable(tf.constant(0.1, shape=[num_classes]))

        out = tf.add(tf.matmul(fc, w), b)
        print("Output {}".format(out.get_shape()))

        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=out, labels=self.input_y)
            self.loss = tf.reduce_mean(losses)

        with tf.name_scope("accuracy"):
            self.y_pred_cls = tf.arg_max(tf.nn.softmax(out), dimension=1)
            self.correct_pred = tf.equal(self.y_pred_cls, tf.argmax(self.input_y, dimension=1))

            self.accuracy = tf.reduce_mean(tf.cast(self.correct_pred, "float"))

# model = Trash_CNN(6, (512, 314, 3), [5, 3, 3], 64)