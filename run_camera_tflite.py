
from my_preprocessing import ssd_preprocessing
import numpy as np
import glob
import cv2
import os
import tqdm
from inference.MultiBBoxDetector import MutliBBoxDetector
from inference.TFLiteMultiBBoxDetectorWithoutPostProcess import TFLiteMutliBBoxDetectorWithoutPostProcess
from pascal_voc_io import PascalVocWriter
import argparse

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Tensorflow Graph Extractor')
    parser.add_argument('--model', type=str, default='mobilenet_v1_ppn_skip', help='vgg / mobilenet_v1 / mobilenet_v1_ppn')
    parser.add_argument('--image_size', type=int, default=300, help='')

    args = parser.parse_args()

    COLORS = ((128, 128, 128), (0, 255, 0), (0, 255, 255), (255, 255, 0), (0, 0, 255))
    CLASSES = ('background',
               'index','other')
    detector = TFLiteMutliBBoxDetectorWithoutPostProcess('./workspace/mobilenet_v1_ppn/detection.tflite',
                                                         './workspace/mobilenet_v1_ppn/anchors.txt',
                                                         (300, 300),
                                                         0.3)

    cap = cv2.VideoCapture(1)
    cap.set(3, 1280)
    cap.set(4, 720)
    while(True):

        _,image=cap.read()
        h,w,_=image.shape
        all_bboxes=detector.inference(image)
        # print(all_labels,all_scores,all_bboxes)
        for bbox in all_bboxes:
            ymin = int(bbox._ymin * h + 0.5)
            xmin = int(bbox._xmin * w + 0.5)
            ymax = int(bbox._ymax * h + 0.5)
            xmax = int(bbox._xmax * w + 0.5)
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), COLORS[int(bbox._class_index)], 3)
            title = "%s:%.2f" % (CLASSES[int(bbox._class_index)], bbox._score)
            p3 = (max(xmin, 15), max(ymin, 15) - 7)
            cv2.putText(image, title, p3, cv2.FONT_ITALIC, 0.6, COLORS[int(bbox._class_index)], 2)

        cv2.imshow('SSD', image)
        if cv2.waitKey(1)==27:
            break

    cv2.destroyAllWindows()





