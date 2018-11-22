import argparse
import logging
import os
import tensorflow as tf
from tensorflow.python.tools import freeze_graph
from net import ssd_net
from  net.mobilenet_v1_backbone import MobileNetV1Backbone
from net.mobilenet_v1_ppn_backbone import MobileNetV1PPNBackbone
from utility import anchor_manipulator
from object_detection import exporter,export_tflite_ssd_graph_lib
from object_detection.builders import post_processing_builder
from object_detection.protos import post_processing_pb2
import numpy as np


logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

def center2point(center_y, center_x, height, width):
    return center_y - height / 2., center_x - width / 2., center_y + height / 2., center_x + width / 2.,

def point2center(ymin, xmin, ymax, xmax):
    height, width = (ymax - ymin), (xmax - xmin)
    return ymin + height / 2., xmin + width / 2., height, width

def encode_anchors(all_anchors):
    list_anchors_ymin = []
    list_anchors_xmin = []
    list_anchors_ymax = []
    list_anchors_xmax = []
    tiled_allowed_borders = []
    for ind, anchor in enumerate(all_anchors):
        anchors_ymin_, anchors_xmin_, anchors_ymax_, anchors_xmax_ = center2point(anchor[0], anchor[1], anchor[2],
                                                                                       anchor[3])

        list_anchors_ymin.append(tf.reshape(anchors_ymin_, [-1]))
        list_anchors_xmin.append(tf.reshape(anchors_xmin_, [-1]))
        list_anchors_ymax.append(tf.reshape(anchors_ymax_, [-1]))
        list_anchors_xmax.append(tf.reshape(anchors_xmax_, [-1]))

    anchors_ymin = tf.concat(list_anchors_ymin, 0, name='concat_ymin')
    anchors_xmin = tf.concat(list_anchors_xmin, 0, name='concat_xmin')
    anchors_ymax = tf.concat(list_anchors_ymax, 0, name='concat_ymax')
    anchors_xmax = tf.concat(list_anchors_xmax, 0, name='concat_xmax')

    anchor_cy, anchor_cx, anchor_h, anchor_w = point2center(anchors_ymin, anchors_xmin, anchors_ymax, anchors_xmax)

    num_anchors = anchor_cy.get_shape().as_list()

    with tf.Session() as sess:
        y_out, x_out, h_out, w_out = sess.run([anchor_cy, anchor_cx, anchor_h, anchor_w])
    encoded_anchors = tf.constant(
        np.transpose(np.stack((y_out, x_out, h_out, w_out))),
        dtype=tf.float32,
        shape=[num_anchors[0], 4])

    return encoded_anchors

def get_network(model_name,input,num_classes,depth_multiplier):
    location_pred, cls_pred=None,None

    if model_name=='mobilenet_v1_ppn':
        anchor_encoder_decoder = anchor_manipulator.AnchorEncoder(allowed_borders=[1.0] * 2,
                                                                  positive_threshold=0.5,
                                                                  ignore_threshold=0.5,
                                                                  prior_scaling=[0.1, 0.1, 0.2, 0.2])

        anchor_creator = anchor_manipulator.AnchorCreator([300, 300],
                                                          layers_shapes=[(19, 19), (10, 10)],
                                                          anchor_scales=[(0.215,), (0.35,)],
                                                          extra_anchor_scales=[(0.275,), (0.418,)],
                                                          anchor_ratios=[(1., .5), (1., .5)],
                                                          layer_steps=None)
        all_anchors, all_num_anchors_depth, all_num_anchors_spatial = anchor_creator.get_all_anchors()

        with tf.variable_scope('FeatureExtractor'):
            backbone = MobileNetV1PPNBackbone('channels_last',depth_multiplier=depth_multiplier)
            feature_layers = backbone.forward(input, is_training=False)
            # print(feature_layers)
            location_pred, cls_pred = ssd_net.multibox_head(feature_layers, num_classes, all_num_anchors_depth,
                                                        data_format='channels_last')
        with tf.variable_scope('raw_outputs'):
            cls_pred = [tf.reshape(pred, [-1, num_classes]) for pred in cls_pred]
            location_pred = [tf.reshape(pred, [-1, 4]) for pred in location_pred]


            cls_pred = tf.expand_dims(tf.concat(cls_pred, axis=0), 0)
            location_pred = tf.expand_dims(tf.concat(location_pred, axis=0),0, name='box_encodings')

            # score_convert_fn_=post_processing_builder._build_score_converter(post_processing_pb2.PostProcessing.SIGMOID,1.0)
            cls_pred=tf.nn.softmax(cls_pred)
            num_anchors = tf.shape(cls_pred)[1]
            cls_pred=tf.slice(cls_pred,[0,0,0],[1,num_anchors,num_classes-1])

            tf.identity(cls_pred, name='class_predictions')

        anchors=encode_anchors(all_anchors)
        tf.identity(anchors,'anchors')


        return cls_pred,location_pred,anchors
    else:
        return None,None,None



if __name__ == '__main__':
    """
    Use this script to just save graph and checkpoint.
    While training, checkpoints are saved. You can test them with this python code.
    """
    parser = argparse.ArgumentParser(description='Tensorflow Graph Extractor')
    parser.add_argument('--model', type=str, default='mobilenet_v1_ppn', help='vgg / mobilenet_v1 / mobilenet_v1_ppn')
    parser.add_argument('--depth_multiplier', type=float, default=0.5, help='')
    parser.add_argument('--num_classes', type=int, default=3, help='')
    parser.add_argument('--checkpoint_path', type=str, default='./logs/mobilenet_ssd/model.ckpt-23355', help='')

    args = parser.parse_args()
    add_postprocessing_op=True


    input_node = tf.placeholder(tf.float32, shape=(1, 300, 300, 3), name='normalized_input_image_tensor')
    net = get_network(args.model,input_node,args.num_classes,args.depth_multiplier)
    # tf.contrib.quantize.create_eval_graph()
    output_dir = './workspace/' + args.model

    with tf.Session() as sess:
        ckpt_path = args.checkpoint_path
        loader = tf.train.Saver()
        try:
            loader.restore(sess, ckpt_path)
        except Exception as e:
            raise Exception('Fail to load model files. \npath=%s\nerr=%s' % (ckpt_path, str(e)))


        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        eval_graph_file=os.path.join(output_dir,'')

        tf.train.write_graph(sess.graph_def, output_dir, 'graph.pb', as_text=True)

        # graph = tf.get_default_graph()
        # for n in tf.get_default_graph().as_graph_def().node:
        #     if 'concat_stage' not in n.name:
        #         continue
        #     print(n.name)

        saver = tf.train.Saver(max_to_keep=100)
        saver.save(sess, './workspace/'+args.model+'/chk', global_step=1)

        summaryWriter = tf.summary.FileWriter('./workspace/'+args.model+'/log/', sess.graph_def)

        input_saver_def = saver.as_saver_def()
        frozen_graph_def = exporter.freeze_graph_with_def_protos(
            input_graph_def=tf.get_default_graph().as_graph_def(),
            input_saver_def=input_saver_def,
            input_checkpoint=ckpt_path,
            output_node_names=','.join([
                'raw_outputs/class_predictions', 'raw_outputs/box_encodings',
                'anchors'
            ]),
            restore_op_name='save/restore_all',
            filename_tensor_name='save/Const:0',
            clear_devices=True,
            output_graph='',
            initializer_nodes='')

        if add_postprocessing_op:
            scale_values={'y_scale':[10.],'x_scale':[10.],'h_scale':[5.],'w_scale':[5.]}
            transformed_graph_def = export_tflite_ssd_graph_lib.append_postprocessing_op(
                frozen_graph_def, max_detections=10, max_classes_per_detection=1,
                nms_score_threshold=[0.01], nms_iou_threshold=[0.5], num_classes=1, scale_values=scale_values)
        else:
            transformed_graph_def=frozen_graph_def

        binary_graph = os.path.join(output_dir, 'tflite_graph.pb')
        with tf.gfile.GFile(binary_graph, 'wb') as f:
            f.write(transformed_graph_def.SerializeToString())

        txt_graph = os.path.join(output_dir, 'tflite_graph.pbtxt')
        with tf.gfile.GFile(txt_graph, 'w') as f:
            f.write(str(transformed_graph_def))


    # os.system('python3 -m tensorflow.python.tools.freeze_graph --input_graph={} --output_graph={} --input_checkpoint={} --output_node_names={}'
    #           .format(os.path.join(output_dir,'graph.pb'),
    #                   os.path.join(output_dir,'exported_freezed_inference_graph.pb'),
    #                   './workspace/' + args.model + '/chk-1',
    #                   'Reshape/class_predictions,Reshape/box_encodings,Reshape/anchors'))



