git clone https://github.com/PaddlePaddle/Paddle-Inference-Demo.git --depth=1
cd Paddle-Inference-Demo/python/cpu/
cd resnet50

# demo 1: cpu-resnet50 单输入模型 oneDNN/OnnxRuntime 预测样例
if [ ! -d resnet50 ]; then
    wget https://paddle-inference-dist.bj.bcebos.com/Paddle-Inference-Demo/resnet50.tgz
    tar xzf resnet50.tgz 
fi

if [ ! -f ILSVRC2012_val_00000247.jpeg ]; then
    wget https://paddle-inference-dist.bj.bcebos.com/inference_demo/python/resnet50/ILSVRC2012_val_00000247.jpeg
fi

# 使用 oneDNN 运行样例
python infer_resnet.py --model_file=./resnet50/inference.pdmodel --params_file=./resnet50/inference.pdiparams

# 使用 OnnxRuntime 预测样例
python infer_resnet.py --model_file=./resnet50/inference.pdmodel --params_file=./resnet50/inference.pdiparams --use_onnxruntime=1

# demo 2: cpu-yolov3 多输入模型 oneDNN/OnnxRuntime 预测样例
cd ../yolov3

if [ ! -d yolov3_r50vd_dcn_270e_coco ]; then
    wget https://paddle-inference-dist.bj.bcebos.com/Paddle-Inference-Demo/yolov3_r50vd_dcn_270e_coco.tgz
    tar xzf yolov3_r50vd_dcn_270e_coco.tgz
fi

if [ ! -f kite.jpg ]; then
    wget https://paddle-inference-dist.bj.bcebos.com/inference_demo/images/kite.jpg
fi

# 使用 oneDNN 运行样例
python infer_yolov3.py --model_file=./yolov3_r50vd_dcn_270e_coco/model.pdmodel --params_file=./yolov3_r50vd_dcn_270e_coco/model.pdiparams

# 使用 OnnxRuntime 预测样例
# python infer_yolov3.py --model_file=./yolov3_r50vd_dcn_270e_coco/model.pdmodel --params_file=./yolov3_r50vd_dcn_270e_coco/model.pdiparams --use_onnxruntime=1

# demo 3: advanced-share_external_data 运行 share_external_data 运行案例
cd ../../advanced/share_external_data/

if [ ! -f ILSVRC2012_val_00000247.jpeg ]; then
    wget https://paddle-inference-dist.bj.bcebos.com/inference_demo/python/resnet50/ILSVRC2012_val_00000247.jpeg
fi

python infer_share_external_data.py --model_file=../../cpu/resnet50/resnet50/inference.pdmodel --params_file=../../cpu/resnet50/resnet50/inference.pdiparams

# demo 4: advanced-multi_thread: 运行多线程运行样例
cd ../multi_thread/

if [ ! -f ILSVRC2012_val_00000247.jpeg ]; then
    wget https://paddle-inference-dist.bj.bcebos.com/inference_demo/python/resnet50/ILSVRC2012_val_00000247.jpeg
fi

python threads_demo.py --model_file=../../cpu/resnet50/resnet50/inference.pdmodel --params_file=../../cpu/resnet50/resnet50/inference.pdiparams --thread_num=2

# demo 5: advanced-custom_operator 自定义算子执行
cd ../custom_operator/
python train_and_infer.py

# demo 6: gpu-resnet50 单输入模型 原生gpu/gpu混合精度推理/Trt_fp32/Trt_fp16/Trt_int8/Trt_dynamic_shape 预测样例

cd ../../gpu/resnet50/

if [ ! -f ILSVRC2012_val_00000247.jpeg ]; then
    wget https://paddle-inference-dist.bj.bcebos.com/inference_demo/python/resnet50/ILSVRC2012_val_00000247.jpeg
fi

# 使用原生 GPU 运行样例
python infer_resnet.py --model_file=../../cpu/resnet50/resnet50/inference.pdmodel --params_file=../../cpu/resnet50/resnet50/inference.pdiparams

# 使用GPU 混合精度推理 运行样例
python infer_resnet.py --model_file=../../cpu/resnet50/resnet50/inference.pdmodel --params_file=../../cpu/resnet50/resnet50/inference.pdiparams --run_mode=gpu_fp16

# 使用 Trt Fp32 运行样例
python infer_resnet.py --model_file=../../cpu/resnet50/resnet50/inference.pdmodel --params_file=../../cpu/resnet50/resnet50/inference.pdiparams --run_mode=trt_fp32

# 使用 Trt Fp16 运行样例
python infer_resnet.py --model_file=../../cpu/resnet50/resnet50/inference.pdmodel --params_file=../../cpu/resnet50/resnet50/inference.pdiparams --run_mode=trt_fp16

# 使用 Trt Int8 运行样例
python infer_resnet.py --model_file=../../cpu/resnet50/resnet50/inference.pdmodel --params_file=../../cpu/resnet50/resnet50/inference.pdiparams --run_mode=trt_int8

python infer_resnet.py --model_file=../../cpu/resnet50/resnet50/inference.pdmodel --params_file=../../cpu/resnet50/resnet50/inference.pdiparams --run_mode=trt_int8

# 使用 Try dynamic shape 运行样例
# python infer_resnet.py --model_file=../../cpu/resnet50/resnet50/inference.pdmodel --params_file=../../cpu/resnet50/resnet50/inference.pdiparams --run_mode=trt_fp32 --use_dynamic_shape=1

# demo 7: gpu-yolov3 多输入模型 原生gpu/gpu混合精度推理/Trt_fp32/Trt_fp16/Trt_int8/Trt_dynamic_shape 预测样例
cd ../yolov3/

# download data
if [ ! -f kite.jpg ]; then
    wget https://paddle-inference-dist.bj.bcebos.com/inference_demo/images/kite.jpg
fi

# 使用原生 GPU 运行样例
python infer_yolov3.py --model_file=../../cpu/yolov3/yolov3_r50vd_dcn_270e_coco/model.pdmodel --params_file=../../cpu/yolov3/yolov3_r50vd_dcn_270e_coco/model.pdiparams

# 使用 Trt Fp32 运行样例
python infer_yolov3.py --model_file=../../cpu/yolov3/yolov3_r50vd_dcn_270e_coco/model.pdmodel --params_file=../../cpu/yolov3/yolov3_r50vd_dcn_270e_coco/model.pdiparams --run_mode=trt_fp32

# 使用 Trt Fp16 运行样例
python infer_yolov3.py --model_file=../../cpu/yolov3/yolov3_r50vd_dcn_270e_coco/model.pdmodel --params_file=../../cpu/yolov3/yolov3_r50vd_dcn_270e_coco/model.pdiparams --run_mode=trt_fp16

# 使用 Trt Int8 运行样例
python infer_yolov3.py --model_file=../../cpu/yolov3/yolov3_r50vd_dcn_270e_coco/model.pdmodel --params_file=../../cpu/yolov3/yolov3_r50vd_dcn_270e_coco/model.pdiparams --run_mode=trt_int8

python infer_yolov3.py --model_file=../../cpu/yolov3/yolov3_r50vd_dcn_270e_coco/model.pdmodel --params_file=../../cpu/yolov3/yolov3_r50vd_dcn_270e_coco/model.pdiparams --run_mode=trt_int8

# demo 8: gpu-tuned_dynamic_shape Trt动态shape自动推导 预测样例 使用 Paddle-TRT TunedDynamicShape 能力
cd ../tuned_dynamic_shape/

# 首先需要针对业务数据进行离线 tune，来获取计算图中所有中间 tensor 的 shape 范围，并将其存储在 config 中配置的shape_range_info.pbtxt 文件中

python infer_tune.py --model_file=../../cpu/resnet50/resnet50/inference.pdmodel --params_file=../../cpu/resnet50/resnet50/inference.pdiparams --tune 1

# 有了离线 tune 得到的 shape 范围信息后，您可以使用该文件自动对所有的 trt 子图设置其输入的 shape 范围。
python infer_tune.py --model_file=../../cpu/resnet50/resnet50/inference.pdmodel --params_file=../../cpu/resnet50/resnet50/inference.pdiparams --use_gpu=1 --use_trt 1 --tuned_dynamic_shape 1

# demo 9: 基于ELMo的LAC分词预测样例
cd ../../mixed/ELMo/

if [ ! -d elmo ]; then
    wget https://paddle-inference-dist.bj.bcebos.com/Paddle-Inference-Demo/elmo.tgz
    tar xzf elmo.tgz 
fi

if [ ! -d elmo_data ]; then
    wget https://paddle-inference-dist.bj.bcebos.com/inference_demo/python/elmo/elmo_data.tgz
    tar xzf elmo_data.tgz
fi

python infer.py

# demo 10: 口罩检测预测样例
cd ../mask_detection/

# demo 11: LSTM INT8 prediction example on X86 Linux
cd ../x86_lstm_demo/

# 1. download model FP32 model for post-training quantization
if [ ! -d lstm_fp32_model ]; then
    wget https://paddle-inference-dist.bj.bcebos.com/lstm/lstm_fp32_model.tar.gz
    tar xzf lstm_fp32_model.tar.gz
fi

# 2. download model quant-aware model
if [ ! -d lstm_quant ]; then
    wget https://paddle-inference-dist.bj.bcebos.com/lstm/lstm_quant.tar.gz
    tar xzf lstm_quant.tar.gz
fi

# 3. download data
if [ ! -f quant_lstm_input_data ]; then
    wget https://paddle-inference-dist.bj.bcebos.com/int8/unittest_model_data/quant_lstm_input_data.tar.gz
    tar xzf quant_lstm_input_data.tar.gz
fi

# 4. download save_quant_model file
if [ ! -f save_quant_model.py ]; then
    wget https://raw.githubusercontent.com/PaddlePaddle/Paddle/develop/python/paddle/fluid/contrib/slim/tests/save_quant_model.py
fi

# 1 thread
# run fp32 model
python model_test.py --model_path=./lstm_fp32_model --data_path=./quant_lstm_input_data --use_analysis=False --num_threads=1

# run ptq int8 model
python model_test.py --model_path=./lstm_fp32_model --data_path=./quant_lstm_input_data --use_ptq=True --num_threads=1

# save quant2 int8 model
python save_quant_model.py --quant_model_path=./lstm_quant --int8_model_save_path=./quant_saved_model_int8

# run quant2 int8 model
python model_test.py --model_path=./quant_saved_model_int8 --data_path=./quant_lstm_input_data --use_analysis=True --num_threads=1

# 4 thread
# run fp32 model
python model_test.py --model_path=./lstm_fp32_model --data_path=./quant_lstm_input_data --use_analysis=False --num_threads=4

# run ptq int8 model
python model_test.py --model_path=./lstm_fp32_model --data_path=./quant_lstm_input_data --use_ptq=True --num_threads=4

# save quant2 int8 model
python save_quant_model.py --quant_model_path=./lstm_quant --int8_model_save_path=./quant_saved_model_int8

# run quant2 int8 model
python model_test.py --model_path=./quant_saved_model_int8 --data_path=./quant_lstm_input_data --use_analysis=True --num_threads=4