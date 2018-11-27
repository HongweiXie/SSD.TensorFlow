import tensorflow as tf
import logging
import cv2
import numpy as np
import math

#0.0495828
#169
class BBox(object):
    def __init__(self,class_index,score,bbox,anchor):
        self._class_index=class_index
        self._score=score
        bbox=self.decode_bbox(bbox, anchor, 0.0495828, 169, 10, 5)
        self._ymin = max(0,bbox[0])
        self._xmin = max(0,bbox[1])
        self._ymax = min(1,bbox[2])
        self._xmax = min(1,bbox[3])


    def decode_bbox(self, bbox, anchor, q_scale, q_z, xy_scale, hw_scale):
        def dequnt(x):
            return q_scale * (float(x) - q_z)

        def decode(bbox):
            y,x,h,w=bbox
            anchor_y,anchor_x,anchor_h,anchor_w=anchor
            y_center=y/xy_scale*anchor_h+anchor_y
            x_center=x/xy_scale*anchor_w+anchor_x
            half_h=0.5*math.exp(h/hw_scale)*anchor_h
            half_w=0.5*math.exp(w/hw_scale)*anchor_w
            ymin=y_center-half_h
            xmin=x_center-half_w
            ymax=y_center+half_h
            xmax=x_center+half_w
            return ymin,xmin,ymax,xmax

        bbox=np.asarray(list(map(dequnt,bbox)))
        bbox=decode(bbox)
        return bbox

    def area(self):
        return (self._xmax-self._xmin)*(self._ymax-self._ymin)




class TFLiteMutliBBoxDetectorWithoutPostProcess(object):
    def __init__(self, tflite_model_file,anchors_file,image_resize_size,threshold=0.01):
        self._logger = logging.getLogger('MultiBBoxDetector')
        self._logger.setLevel(logging.INFO)

        self._logger.info('loading graph from %s' % (tflite_model_file))
        self._image_resize_size=image_resize_size
        self._threshold=threshold

        self.interpreter = tf.contrib.lite.Interpreter(model_path=str(tflite_model_file))
        self.interpreter.allocate_tensors()
        self.input_index = self.interpreter.get_input_details()[0]['index']
        self.outputs = self.interpreter.get_output_details()
        self.class_predictions=self.outputs[0]['index']
        self.bbox_predictions=self.outputs[1]['index']
        self.anchors=self.load_anchors(anchors_file)



    def load_anchors(self,anchors_file):
        result=[]
        with open(anchors_file,'r') as f:
            lines=f.read().split('\n')
            for line in lines:
                if len(line)<10:
                    continue
                values=np.asarray(list(map(float,line.split(' '))))
                result.append(values)
        return np.asarray(result)

    def inference(self,image):
        image=cv2.resize(image,self._image_resize_size)
        # image = image.astype(np.float32)
        # image = image - 127.5
        # image = image * 0.007843

        self.interpreter.set_tensor(self.input_index, [image])
        self.interpreter.invoke()
        scores=self.interpreter.get_tensor(self.class_predictions)
        bboxes=self.interpreter.get_tensor(self.bbox_predictions)
        selected_bboxes=self.select_bbox(scores[0],bboxes[0])
        return self.nms(selected_bboxes,0.45)


    def select_bbox(self,scores,bboxes):
        threshold = self._threshold
        num_classes=len(scores[0])
        selected_bbox=[]
        for i in range(num_classes):
            selected_bbox.append([])

        for k in range(len(scores)):
            score_tuple = scores[k]
            for i in range(1,num_classes):
                score=score_tuple[i]/255.
                if score>=threshold and score_tuple[i]>score_tuple[0]:
                    bbox=BBox(i,score,bboxes[k],self.anchors[k])
                    if bbox._xmax>bbox._xmin and bbox._ymax>bbox._ymin:
                        selected_bbox[i].append(bbox)

        return selected_bbox

    def iou(self,b1, b2):
        xA = max(b1._xmin, b2._xmin)
        yA = max(b1._ymin, b2._ymin)
        xB = min(b1._xmax, b2._xmax)
        yB = min(b1._ymax, b2._ymax)

        if xB < xA or yB < yA:
            return 0
        # compute the area of intersection rectangle
        interArea = (xB - xA) * (yB - yA)

        # compute the area of both the prediction and ground-truth
        # rectangles
        boxAArea = b1.area()
        boxBArea = b2.area()

        # compute the intersection over union by taking the intersection
        # area and dividing it by the sum of prediction + ground-truth
        # areas - the interesection area
        iou = interArea / float(boxAArea + boxBArea - interArea)

        # return the intersection over union value
        return iou

    def nms(self,selected_bboxes, iou_threshold):
        result = []
        for bboxes in selected_bboxes:
            if len(bboxes) == 0:
                continue
            bboxes = sorted(bboxes, key=lambda x: x._score, reverse=True)
            candidates = []
            for i in range(len(bboxes)):
                is_candidates = True
                for j in range(len(candidates)):
                    if self.iou(candidates[j], bboxes[i]) > iou_threshold:
                        is_candidates = False
                        break
                if is_candidates:
                    candidates.append(bboxes[i])
                    result.append(bboxes[i])
        return result

