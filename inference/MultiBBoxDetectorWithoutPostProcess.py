import tensorflow as tf
import logging
import cv2


class MutliBBoxDetectorWithoutPostProcess(object):
    def __init__(self, graph_path,image_resize_size):
        self._logger = logging.getLogger('MultiBBoxDetector')
        self._logger.setLevel(logging.INFO)

        self._logger.info('loading graph from %s' % (graph_path))
        self._image_resize_size=image_resize_size
        with tf.gfile.GFile(graph_path, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())

            self._graph = tf.get_default_graph()
            tf.import_graph_def(graph_def, name='MultiBBoxDetector')
            self._persistent_sess = tf.Session(graph=self._graph)
            self._tensor_image = self._graph.get_tensor_by_name('MultiBBoxDetector/normalized_input_image_tensor:0')
            self._tensor_scores = self._graph.get_tensor_by_name('MultiBBoxDetector/raw_outputs/class_predictions:0')
            self._tensor_bboxes = self._graph.get_tensor_by_name('MultiBBoxDetector/raw_outputs/box_encodings:0')


    def inference(self,image):
        image=cv2.resize(image,self._image_resize_size)
        image = image.astype(float)
        image = image - 127.5
        image = image * 0.007843
        all_scores,all_bboxes=self._persistent_sess.run([self._tensor_scores,self._tensor_bboxes],
                                  feed_dict={self._tensor_image:[image]})
        return all_scores,all_bboxes