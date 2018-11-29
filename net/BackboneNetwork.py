import tensorflow as tf
slim = tf.contrib.slim
DEFAULT_PADDING = 'SAME'
_init_xavier = tf.contrib.layers.xavier_initializer()
_init_zero = slim.init_ops.zeros_initializer()
_l2_regularizer_00004 = tf.contrib.layers.l2_regularizer(0.00004)

class BackboneNetwork(object):

    def forward(self,inputs, is_training=False):
        pass

    def upsample(self, input, factor, name):
        return tf.image.resize_bilinear(input, [int(input.get_shape()[1]) * factor, int(input.get_shape()[2]) * factor], name=name)

    def separable_conv(self, input, k_h, k_w, c_o, stride, name, relu=True, set_bias=True,is_training=True):
        with slim.arg_scope([slim.batch_norm], decay=0.97,
                                epsilon=0.001,
                                scale=True,
                                center=True,is_training=is_training):
            with slim.arg_scope([slim.conv2d, slim.separable_conv2d],
                                activation_fn=tf.nn.relu6,
                                normalizer_fn=slim.batch_norm,
                                weights_regularizer=_l2_regularizer_00004,
                                weights_initializer=_init_xavier
                                ):
                output = slim.separable_convolution2d(input,
                                                      num_outputs=None,
                                                      stride=stride,
                                                      depth_multiplier=1.0,
                                                      kernel_size=[k_h, k_w],
                                                      padding=DEFAULT_PADDING,
                                                      scope=name + '_depthwise')

                output = slim.convolution2d(output,
                                            c_o,
                                            stride=1,
                                            kernel_size=[1, 1],
                                            scope=name + '_pointwise')

        return output

    def convb(self, input, k_h, k_w, c_o, stride, name, relu=True, set_bias=True, set_tanh=False,is_training=True):
        with slim.arg_scope([slim.batch_norm], decay=0.999, is_training=is_training):
            output = slim.convolution2d(input, c_o, kernel_size=[k_h, k_w],
                                        stride=stride,
                                        normalizer_fn=slim.batch_norm,
                                        weights_regularizer=None,
                                        weights_initializer=_init_xavier,
                                        biases_initializer=_init_zero if set_bias else None,
                                        trainable=is_training,
                                        activation_fn=tf.nn.relu6 if relu else None,
                                        scope=name)
        return output

    def max_pool(self, input, k_h, k_w, s_h, s_w, name, padding=DEFAULT_PADDING):
        return tf.nn.max_pool(input,
                              ksize=[1, k_h, k_w, 1],
                              strides=[1, s_h, s_w, 1],
                              padding=padding,
                              name=name)