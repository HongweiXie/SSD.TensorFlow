import tensorflow as tf
import logging
import cv2
import numpy as np

class TFLiteMutliBBoxDetectorWithoutPostProcess(object):
    def __init__(self, tflite_model_file,image_resize_size):
        self._logger = logging.getLogger('MultiBBoxDetector')
        self._logger.setLevel(logging.INFO)

        self._logger.info('loading graph from %s' % (tflite_model_file))
        self._image_resize_size=image_resize_size

        self.interpreter = tf.contrib.lite.Interpreter(model_path=str(tflite_model_file))
        self.interpreter.allocate_tensors()
        self.input_index = self.interpreter.get_input_details()[0]['index']
        self.outputs = self.interpreter.get_output_details()
        self.class_predictions=self.outputs[0]['index']
        self.bbox_predictions=self.outputs[1]['index']



    def inference(self,image):
        image=cv2.resize(image,self._image_resize_size)
        image = image.astype(np.float32)
        image = image - 127.5
        image = image * 0.007843

        self.interpreter.set_tensor(self.input_index, [image])
        self.interpreter.invoke()
        classes=self.interpreter.get_tensor(self.class_predictions)
        bboxes=self.interpreter.get_tensor(self.bbox_predictions)
        print(classes)
        print(bboxes)
        return classes,bboxes