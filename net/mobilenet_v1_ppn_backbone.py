from __future__ import absolute_import
import tensorflow as tf
from nets import mobilenet_v1
from net.BackboneNetwork import BackboneNetwork
from object_detection.models import feature_map_generators
from object_detection import exporter
slim = tf.contrib.slim

_init_xavier = tf.contrib.layers.xavier_initializer()
_init_norm = tf.truncated_normal_initializer(stddev=0.01)
_init_zero = slim.init_ops.zeros_initializer()
_l2_regularizer_00004 = tf.contrib.layers.l2_regularizer(0.00004)

class MobileNetV1PPNBackbone(BackboneNetwork):
    def __init__(self, data_format='channels_first',depth_multiplier=0.5):
        super(MobileNetV1PPNBackbone, self).__init__()
        self._data_format = data_format
        self._depth_multiplier=depth_multiplier


    def forward(self, inputs, is_training=False):
        features=[]
        with tf.variable_scope('MobilenetV1') as scope:
            with slim.arg_scope([slim.batch_norm],
                                decay=0.97,
                                epsilon=0.001,
                                scale=True,
                                center=True
                                ):
                with slim.arg_scope(
                        mobilenet_v1.mobilenet_v1_arg_scope(
                            is_training=is_training, regularize_depthwise=True)):
                    with slim.arg_scope([slim.conv2d, slim.separable_conv2d],
                                        activation_fn=tf.nn.relu6,
                                        normalizer_fn=slim.batch_norm,
                                        weights_regularizer=_l2_regularizer_00004,
                                        weights_initializer=_init_xavier
                                        ):
                        _, image_features = mobilenet_v1.mobilenet_v1_base(
                            inputs,
                            final_endpoint='Conv2d_11_pointwise',
                            min_depth=8,
                            depth_multiplier=self._depth_multiplier,
                            use_explicit_padding=False,
                            scope=scope)

            with slim.arg_scope([slim.batch_norm],
                                decay=0.97,
                                epsilon=0.001,
                                scale=True,
                                center=True
                                ):
                with slim.arg_scope([slim.conv2d, slim.separable_conv2d],
                                    activation_fn=tf.nn.relu6,
                                    normalizer_fn=slim.batch_norm,
                                    weights_regularizer=_l2_regularizer_00004,
                                    weights_initializer=_init_xavier
                                    ):
                    input_keys={
                            'image_features': image_features['Conv2d_11_pointwise']
                        }
                    print(list(input_keys.keys())[0])
                    feature_maps = feature_map_generators.pooling_pyramid_feature_maps(
                        base_feature_map_depth=0,
                        num_layers=2,
                        image_features={
                            'image_features': image_features['Conv2d_11_pointwise']
                        })
                    for key in feature_maps:
                        features.append(feature_maps[key])

        return features


if __name__ == '__main__':
    #300x300x0.5=473.26M flops 300x300x0.25=130.79M flops
    input=tf.placeholder(tf.float32,shape=(1,300,300,3),name='image')
    feature_extractor=MobileNetV1PPNBackbone(depth_multiplier=0.25)
    features=feature_extractor.forward(input,True)
    exporter.profile_inference_graph(tf.get_default_graph())
    print(features)

