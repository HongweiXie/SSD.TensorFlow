import argparse
import logging
import os
import tensorflow as tf
from tensorflow.python.tools import freeze_graph
from net import ssd_net
from  net.mobilenet_v1_backbone import MobileNetV1Backbone
from net.mobilenet_v1_ppn_backbone import MobileNetV1PPNBackbone
from net.mobilenet_v1_ppn_skip_backbone import MobileNetV1PPNSkipBackbone
from net.mobilenet_v1_ppn_branch_backbone import MobileNetV1PPNBranchBackbone
from utility import anchor_manipulator
from object_detection import exporter

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
def select_bboxes(scores_pred, bboxes_pred, num_classes, select_threshold):
    selected_bboxes = {}
    selected_scores = {}
    with tf.name_scope('select_bboxes', [scores_pred, bboxes_pred]):
        for class_ind in range(1, num_classes):
            class_scores = scores_pred[:, class_ind]
            select_mask = class_scores > select_threshold

            select_mask = tf.cast(select_mask, tf.float32)
            selected_bboxes[class_ind] = tf.multiply(bboxes_pred, tf.expand_dims(select_mask, axis=-1))
            selected_scores[class_ind] = tf.multiply(class_scores, select_mask)

    return selected_bboxes, selected_scores

def clip_bboxes(ymin, xmin, ymax, xmax, name):
    with tf.name_scope(name, 'clip_bboxes', [ymin, xmin, ymax, xmax]):
        ymin = tf.maximum(ymin, 0.)
        xmin = tf.maximum(xmin, 0.)
        ymax = tf.minimum(ymax, 1.)
        xmax = tf.minimum(xmax, 1.)

        ymin = tf.minimum(ymin, ymax)
        xmin = tf.minimum(xmin, xmax)

        return ymin, xmin, ymax, xmax

def filter_bboxes(scores_pred, ymin, xmin, ymax, xmax, min_size, name):
    with tf.name_scope(name, 'filter_bboxes', [scores_pred, ymin, xmin, ymax, xmax]):
        width = xmax - xmin
        height = ymax - ymin

        filter_mask = tf.logical_and(width > min_size, height > min_size)

        filter_mask = tf.cast(filter_mask, tf.float32)
        return tf.multiply(ymin, filter_mask), tf.multiply(xmin, filter_mask), \
                tf.multiply(ymax, filter_mask), tf.multiply(xmax, filter_mask), tf.multiply(scores_pred, filter_mask)

def sort_bboxes(scores_pred, ymin, xmin, ymax, xmax, keep_topk, name):
    with tf.name_scope(name, 'sort_bboxes', [scores_pred, ymin, xmin, ymax, xmax]):
        cur_bboxes = tf.shape(scores_pred)[0]
        scores, idxes = tf.nn.top_k(scores_pred, k=tf.minimum(keep_topk, cur_bboxes), sorted=True)

        ymin, xmin, ymax, xmax = tf.gather(ymin, idxes), tf.gather(xmin, idxes), tf.gather(ymax, idxes), tf.gather(xmax, idxes)

        paddings_scores = tf.expand_dims(tf.stack([0, tf.maximum(keep_topk-cur_bboxes, 0)], axis=0), axis=0)

        return tf.pad(ymin, paddings_scores, "CONSTANT"), tf.pad(xmin, paddings_scores, "CONSTANT"),\
                tf.pad(ymax, paddings_scores, "CONSTANT"), tf.pad(xmax, paddings_scores, "CONSTANT"),\
                tf.pad(scores, paddings_scores, "CONSTANT")

def nms_bboxes(scores_pred, bboxes_pred, nms_topk, nms_threshold, name):
    with tf.name_scope(name, 'nms_bboxes', [scores_pred, bboxes_pred]):
        idxes = tf.image.non_max_suppression(bboxes_pred, scores_pred, nms_topk, nms_threshold)
        return tf.gather(scores_pred, idxes,name='nms_scores'), tf.gather(bboxes_pred, idxes,name='nms_bboxes')


def parse_by_class(cls_pred, bboxes_pred, num_classes, select_threshold, min_size, keep_topk, nms_topk, nms_threshold):
    with tf.name_scope('select_bboxes', [cls_pred, bboxes_pred]):
        scores_pred = tf.nn.softmax(cls_pred)
        selected_bboxes, selected_scores = select_bboxes(scores_pred, bboxes_pred, num_classes, select_threshold)
        for class_ind in range(1, num_classes):
            ymin, xmin, ymax, xmax = tf.unstack(selected_bboxes[class_ind], 4, axis=-1)
            #ymin, xmin, ymax, xmax = tf.split(selected_bboxes[class_ind], 4, axis=-1)
            #ymin, xmin, ymax, xmax = tf.squeeze(ymin), tf.squeeze(xmin), tf.squeeze(ymax), tf.squeeze(xmax)
            ymin, xmin, ymax, xmax = clip_bboxes(ymin, xmin, ymax, xmax, 'clip_bboxes_{}'.format(class_ind))
            ymin, xmin, ymax, xmax, selected_scores[class_ind] = filter_bboxes(selected_scores[class_ind],
                                                ymin, xmin, ymax, xmax, min_size, 'filter_bboxes_{}'.format(class_ind))
            ymin, xmin, ymax, xmax, selected_scores[class_ind] = sort_bboxes(selected_scores[class_ind],
                                                ymin, xmin, ymax, xmax, keep_topk, 'sort_bboxes_{}'.format(class_ind))
            selected_bboxes[class_ind] = tf.stack([ymin, xmin, ymax, xmax], axis=-1)
            selected_scores[class_ind], selected_bboxes[class_ind] = nms_bboxes(selected_scores[class_ind], selected_bboxes[class_ind], nms_topk, nms_threshold, 'nms_bboxes_{}'.format(class_ind))

        return selected_bboxes, selected_scores

def get_network(model_name,input,input_size,num_classes,depth_multiplier):
    location_pred, cls_pred=None,None

    if str(model_name).startswith('mobilenet_v1_ppn'):
        anchor_encoder_decoder = anchor_manipulator.AnchorEncoder(allowed_borders=[1.0] * 2,
                                                                  positive_threshold=0.5,
                                                                  ignore_threshold=0.5,
                                                                  prior_scaling=[0.1, 0.1, 0.2, 0.2])

        if model_name=='mobilenet_v1_ppn_skip' or model_name=='mobilenet_v1_ppn_branch':
            feat_l1_shape=(int(input_size/8.+0.99),int(input_size/8.+0.99))
            feat_l2_shape=(int(input_size/16.+0.99),int(input_size/16.+0.99))
        else:
            feat_l1_shape = (int(input_size / 16. + 0.99), int(input_size / 16. + 0.99))
            feat_l2_shape = (int(input_size / 32. + 0.99), int(input_size / 32. + 0.99))

        anchor_creator = anchor_manipulator.AnchorCreator([input_size, input_size],
                                                          layers_shapes=[feat_l1_shape, feat_l2_shape],
                                                          anchor_scales=[(0.215,), (0.35,)],
                                                          extra_anchor_scales=[(0.275,), (0.418,)],
                                                          anchor_ratios=[(1., .5), (1., .5)],
                                                          layer_steps=None)
        all_anchors, all_num_anchors_depth, all_num_anchors_spatial = anchor_creator.get_all_anchors()

        with tf.variable_scope('FeatureExtractor'):
            if model_name=='mobilenet_v1_ppn':
                backbone = MobileNetV1PPNBackbone('channels_last',depth_multiplier=depth_multiplier)
            elif model_name=='mobilenet_v1_ppn_skip':
                backbone = MobileNetV1PPNSkipBackbone('channels_last', depth_multiplier=depth_multiplier)
            else:
                backbone = MobileNetV1PPNBranchBackbone('channels_last', depth_multiplier=depth_multiplier)
            feature_layers = backbone.forward(input, is_training=False)
            # print(feature_layers)
            location_pred, cls_pred = ssd_net.multibox_head(feature_layers, num_classes, all_num_anchors_depth,
                                                        data_format='channels_last')
        with tf.variable_scope('PostProcess'):

            with tf.variable_scope('Reshape'):
                cls_pred = [tf.reshape(pred, [-1, num_classes]) for pred in cls_pred]
                location_pred = [tf.reshape(pred, [-1, 4]) for pred in location_pred]

                cls_pred = tf.concat(cls_pred, axis=0)
                location_pred = tf.concat(location_pred, axis=0)

            with tf.variable_scope('Decoder'),tf.device('/cpu:0'):
                num_anchors_per_layer = []
                for ind in range(len(all_anchors)):
                    num_anchors_per_layer.append(all_num_anchors_depth[ind] * all_num_anchors_spatial[ind])
                bboxes_pred = anchor_encoder_decoder.ext_decode_all_anchors(location_pred,all_anchors,all_num_anchors_depth,all_num_anchors_spatial)
                bboxes_pred = tf.concat(bboxes_pred, axis=0)

                selected_bboxes, selected_scores = parse_by_class(cls_pred, bboxes_pred,
                                                                  num_classes,
                                                                  select_threshold=0.01,
                                                                  min_size=0.03,
                                                                  keep_topk=400, nms_topk=200,
                                                                  nms_threshold=0.45)
                labels_list = []
                scores_list = []
                bboxes_list = []
                for k, v in selected_scores.items():
                    labels_list.append(tf.ones_like(v, tf.int32) * k)
                    scores_list.append(v)
                    bboxes_list.append(selected_bboxes[k])
                all_labels = tf.concat(labels_list, axis=0,name='all_labels')
                all_scores = tf.concat(scores_list, axis=0,name='all_scores')
                all_bboxes = tf.concat(bboxes_list, axis=0,name='all_bboxes')
        return all_labels,all_scores,all_bboxes
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
    parser.add_argument('--image_size', type=int, default=300, help='')
    parser.add_argument('--quant',type=bool,default=False)

    args = parser.parse_args()
    input_size=args.image_size
    input_node = tf.placeholder(tf.float32, shape=(1, input_size, input_size, 3), name='image')
    net = get_network(args.model,input_node,input_size,args.num_classes,args.depth_multiplier)
    if args.quant:
        tf.contrib.quantize.create_eval_graph()
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
        # exporter.profile_inference_graph(tf.get_default_graph())

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
                'PostProcess/Decoder/all_labels', 'PostProcess/Decoder/all_scores',
                'PostProcess/Decoder/all_bboxes'
            ]),
            restore_op_name='save/restore_all',
            filename_tensor_name='save/Const:0',
            clear_devices=True,
            output_graph='',
            initializer_nodes='')

        binary_graph = os.path.join(output_dir, 'exported_freezed_inference_graph.pb')
        with tf.gfile.GFile(binary_graph, 'wb') as f:
            f.write(frozen_graph_def.SerializeToString())

    # os.system('python3 -m tensorflow.python.tools.freeze_graph --input_graph={} --output_graph={} --input_checkpoint={} --output_node_names={}'
    #           .format(os.path.join(output_dir,'graph.pb'),
    #                   os.path.join(output_dir,'exported_freezed_inference_graph.pb'),
    #                   './workspace/' + args.model + '/chk-1',
    #                   'PostProcess/Decoder/all_labels,PostProcess/Decoder/all_scores,PostProcess/Decoder/all_bboxes'))
    #
