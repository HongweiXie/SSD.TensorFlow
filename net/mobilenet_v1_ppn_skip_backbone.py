from __future__ import absolute_import
import tensorflow as tf
from nets import mobilenet_v1
from object_detection.models import feature_map_generators
from object_detection import exporter
from net.BackboneNetwork import BackboneNetwork
slim = tf.contrib.slim

_init_xavier = tf.contrib.layers.xavier_initializer()
_init_norm = tf.truncated_normal_initializer(stddev=0.01)
_init_zero = slim.init_ops.zeros_initializer()
_l2_regularizer_00004 = tf.contrib.layers.l2_regularizer(0.00004)

class MobileNetV1PPNSkipBackbone(BackboneNetwork):
    def __init__(self, data_format='channels_last',depth_multiplier=0.5):
        super(MobileNetV1PPNSkipBackbone, self).__init__()
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
            # conv2d_3_pool=self.max_pool(image_features['Conv2d_3_pointwise'],2, 2, 2, 2, name='Conv2d_3_pool')
            conv2d_11_upsample=self.upsample(image_features['Conv2d_11_pointwise'],2,name='Conv2d_11_pointwise_upsample')
            # conv2d_7_upsample=self.upsample(image_features['Conv2d_7_pointwise'], 2, name='Conv2d_7_pointwise_upsample')
            features_concat=tf.concat([image_features['Conv2d_5_pointwise'],conv2d_11_upsample],axis=3)
            prefix='Feature_Concat'
            features_concat=self.separable_conv(features_concat,3, 3, min(256,192*4*self._depth_multiplier), 1, name=prefix + '_L_1')
            features_concat = self.separable_conv(features_concat, 3, 3, 256, 1, name=prefix + '_L_2')
            # features_concat = self.separable_conv(features_concat, 3, 3, 256, 1, name=prefix + '_L_3')
            # features_concat = self.convb(features_concat, 1, 1, 256, 1, name=prefix + '_L_4')
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
                    feature_maps = feature_map_generators.pooling_pyramid_feature_maps(
                        base_feature_map_depth=0,
                        num_layers=2,
                        image_features={
                            'image_features': features_concat
                        })
                    for key in feature_maps:
                        features.append(feature_maps[key])

        return features




if __name__ == '__main__':
    input=tf.placeholder(tf.float32,shape=(1,192,192,3),name='image')
    feature_extractor=MobileNetV1PPNSkipBackbone(depth_multiplier=0.25)
    features=feature_extractor.forward(input,True)
    exporter.profile_inference_graph(tf.get_default_graph())
    print(features)

