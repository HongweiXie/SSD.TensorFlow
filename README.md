实现MobilenetSSD

使用方式:
1. 需要python3+tensorflow
2. 用dataset中的脚本先创建tfrecord,输入格式是Pascal-VOC格式
3. train_hand.sh训练  具体参数看train_mobilenet_ssd.py  量化训练加--quant=True
4. 导出pb模型用export_eval_graph.py 导出tflite用export_tflite_eval_graph.py --add_postprocessing_op添加TFlite提供的Custom后处理Op,会只能够在Float版本运行,--quant对应是量化训练的网络
5. 通过run_eval.py run_eval_tflite.py run_eval_tflite_no_postprocess.py eval相应的模型

