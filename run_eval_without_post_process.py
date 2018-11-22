
from my_preprocessing import ssd_preprocessing
import numpy as np
import glob
import cv2
import os
import tqdm
from inference.MultiBBoxDetectorWithoutPostProcess import MutliBBoxDetectorWithoutPostProcess
from pascal_voc_io import PascalVocWriter


if __name__ == '__main__':
    COLORS = ((128, 128, 128), (0, 255, 0), (0, 255, 255), (255, 255, 0), (0, 0, 255))
    CLASSES = ('background',
               'index','other')
    detector=MutliBBoxDetectorWithoutPostProcess('./workspace/mobilenet_v1_ppn/tflite_graph.pb',(300,300))
    input_path='/home/sixd-ailabs/Develop/Human/Hand/diandu/test/taideng/test'
    output_path='/home/sixd-ailabs/Develop/Human/Hand/diandu/test/eval_taideng'
    show=True
    jpg_list=glob.glob(input_path+'/*.jpg')
    for jpg_file in tqdm.tqdm(jpg_list):
        print(jpg_file)
        image=cv2.imread(jpg_file)
        jpg_name= os.path.basename(jpg_file)
        h,w,_=image.shape
        all_scores,all_bboxes=detector.inference(image)
        # print(all_labels,all_scores,all_bboxes)
        # if show:
        #     print(jpg_file)
        # writer=PascalVocWriter('test',jpg_name,image.shape,localImgPath=os.path.join(output_path,jpg_name))
        # for label,score,bbox in zip(all_labels,all_scores,all_bboxes):
        #     if score>0.01:
        #         ymin=int(bbox[0]*h+0.5)
        #         xmin=int(bbox[1]*w+0.5)
        #         ymax=int(bbox[2]*h+0.5)
        #         xmax=int(bbox[3]*w+0.5)
        #         writer.addBndBox(xmin,ymin,xmax,ymax,CLASSES[int(label)],False,score)
        #         if(show):
        #             if(score>0.3):
        #                 cv2.rectangle(image, (xmin, ymin), (xmax, ymax), COLORS[int(label)], 3)
        #                 title = "%s:%.2f" % (CLASSES[int(label)], score)
        #                 p3 = (max(xmin, 15), max(ymin, 15) - 7)
        #                 cv2.putText(image, title, p3, cv2.FONT_ITALIC, 0.6, COLORS[int(label)], 2)
        # writer.save(os.path.join(output_path,jpg_name[:-4]+'.xml'))
        #
        # if(show):
        #     cv2.imshow('SSD', image)
        #     cv2.waitKey(0)
    if show:
        cv2.destroyAllWindows()





