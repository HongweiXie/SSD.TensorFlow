#!/usr/bin/env bash

# /home/sixd-ailabs/Develop/DL/TF/test/tensorflow/bazel-bin/tensorflow/contrib/lite/toco/toco \
# --input_file=./workspace/mobilenet_v1_ppn/tflite_graph.pb \
# --output_file=./workspace/mobilenet_v1_ppn/detection.tflite \
# --input_shapes=1,300,300,3 \
# --input_arrays=image \
# --output_arrays='raw_outputs/class_predictions','raw_outputs/box_encodings','anchors' \
# --inference_type=FLOAT \
# --mean_values=128 \
# --std_values=128 \
# --change_concat_input_ranges=false \
# --allow_custom_ops\
# --default_ranges_min=0 \
# --default_ranges_max=1

 /home/sixd-ailabs/Develop/DL/TF/test/tensorflow/bazel-bin/tensorflow/contrib/lite/toco/toco \
 --input_file=./workspace/mobilenet_v1_ppn/tflite_graph.pb \
 --output_file=./workspace/mobilenet_v1_ppn/detection.tflite \
 --input_shapes=1,300,300,3 \
 --input_arrays=normalized_input_image_tensor \
 --output_arrays='raw_outputs/class_predictions','raw_outputs/box_encodings'\
 --inference_type=FLOAT  \
 --input_data_types=FLOAT  \
 --allow_custom_ops

# toco \
# --output_file=./workspace/mobilenet_v1_ppn/detection.tflite \
# --graph_def_file=./workspace/mobilenet_v1_ppn/tflite_graph.pb \
# --output_format=TFLITE \
# --inference_type=FLOAT  \
# --inference_input_type=FLOAT \
# --input_data_types=FLOAT\
# --input_shapes=1,300,300,3 \
# --input_arrays=normalized_input_image_tensor \
# --output_arrays='TFLite_Detection_PostProcess','TFLite_Detection_PostProcess:1','TFLite_Detection_PostProcess:2','TFLite_Detection_PostProcess:3' \
# --allow_custom_ops