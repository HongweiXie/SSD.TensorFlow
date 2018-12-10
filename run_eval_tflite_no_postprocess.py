
from my_preprocessing import ssd_preprocessing
import numpy as np
import glob
import cv2
import os
import tqdm
from inference.TFLiteMultiBBoxDetector import TFLiteMutliBBoxDetector
from inference.TFLiteMultiBBoxDetectorWithoutPostProcess import TFLiteMutliBBoxDetectorWithoutPostProcess
from pascal_voc_io import PascalVocWriter
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Tensorflow Graph Extractor')
    parser.add_argument('--model', type=str, default='mobilenet_v1_ppn_skip',
                        help='vgg / mobilenet_v1 / mobilenet_v1_ppn')
    parser.add_argument('--input_path', type=str,
                        default='/home/sixd-ailabs/Develop/Human/Hand/diandu/test/chengren_17', help='')
    parser.add_argument('--output_path', type=str,
                        default='/home/sixd-ailabs/Develop/Human/Hand/diandu/test/eval_chengren_17_lr', help='')
    parser.add_argument('--image_size', type=int, default=128, help='')
    parser.add_argument('--show', type=bool, default=False, help='')
    args = parser.parse_args()

    COLORS = ((128, 128, 128), (0, 255, 0), (0, 255, 255), (255, 255, 0), (0, 0, 255))
    CLASSES = ('background',
               'index','other')
    detector=TFLiteMutliBBoxDetectorWithoutPostProcess('./workspace/'+args.model+'/detection.tflite','./workspace/'+args.model+'/anchors.txt',(args.image_size,args.image_size),is_quant=True)
    input_path=args.input_path
    output_path=args.output_path
    # input_path='/home/sixd-ailabs/Downloads/test/test'
    # output_path='/home/sixd-ailabs/Downloads/test/test'
    # input_path='/home/sixd-ailabs/Develop/Human/Hand/diandu/test/taideng/test'
    # output_path='/home/sixd-ailabs/Develop/Human/Hand/diandu/test/eval_taideng'
    # input_path = '/home/sixd-ailabs/Develop/Human/Hand/diandu/test/chengren_17'
    # output_path = '/home/sixd-ailabs/Develop/Human/Hand/diandu/test/eval_chengren_17_lr'
    show=args.show
    jpg_list=glob.glob(input_path+'/*.jpg')
    jpg_list=sorted(jpg_list)
    for jpg_file in tqdm.tqdm(jpg_list):

        image=cv2.imread(jpg_file)
        jpg_name= os.path.basename(jpg_file)
        h,w,_=image.shape
        all_bboxes=detector.inference(image)

        if show:
            print(jpg_file)
        writer=PascalVocWriter('test',jpg_name,image.shape,localImgPath=os.path.join(output_path,jpg_name))
        for bbox in all_bboxes:
            if bbox._score>0.03:
                ymin=int(bbox._ymin*h+0.5)
                xmin=int(bbox._xmin*w+0.5)
                ymax=int(bbox._ymax*h+0.5)
                xmax=int(bbox._xmax*w+0.5)
                writer.addBndBox(xmin,ymin,xmax,ymax,CLASSES[int(bbox._class_index)],False,bbox._score)
                if(show):
                    if(bbox._score>0.03):
                        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), COLORS[int(bbox._class_index)], 3)
                        title = "%s:%.2f" % (CLASSES[int(bbox._class_index)], bbox._score)
                        p3 = (max(xmin, 15), max(ymin, 15) - 7)
                        cv2.putText(image, title, p3, cv2.FONT_ITALIC, 0.6, COLORS[int(bbox._class_index)], 2)
        writer.save(os.path.join(output_path,jpg_name[:-4]+'.xml'))

        if(show):
            cv2.imshow('SSD', image)
            cv2.waitKey(0)
    if show:
        cv2.destroyAllWindows()





