#!/bin/bash
set -x
pwd

git clone https://github.com/PaddlePaddle/Paddle-Inference-Demo.git --depth=1
cd Paddle-Inference-Demo/python/cpu/resnet50

error=0
demo=0
case=0

echo "============= The path of Paddle-Inference-Demo Test failed cases  =============" >> /workspace/Paddle-Yu/Paddle-Inference-Demo/test_result.txt
# 定义 error 计算方法 
count_error() {
    if [ $? -ne 0 ]; then
        error=`expr ${error} + 1`
        echo ${PWD} >> /workspace/Paddle-Yu/Paddle-Inference-Demo/test_result.txt
    fi
    case=`expr ${case} + 1`
}

echo "Python Inference demo:"

echo "demo 1: cpu_resnet50:"
# demo 1: cpu-resnet50 单输入模型 oneDNN/OnnxRuntime 预测样例
demo=`expr ${demo} + 1`

sed -i "s/is not/!=/" infer_resnet.py

if [ ! -d resnet50 ]; then
    wget -q https://paddle-inference-dist.bj.bcebos.com/Paddle-Inference-Demo/resnet50.tgz
    tar xzf resnet50.tgz 
fi

if [ ! -f ILSVRC2012_val_00000247.jpeg ]; then
    wget -q https://paddle-inference-dist.bj.bcebos.com/inference_demo/python/resnet50/ILSVRC2012_val_00000247.jpeg
fi

echo "demo 1.1 resnet50_mkldnn:"
# 使用 oneDNN 运行样例
python infer_resnet.py --model_file=./resnet50/inference.pdmodel --params_file=./resnet50/inference.pdiparams
count_error

echo "demo 1.2 resnet50_onnxruntime:"
# 使用 OnnxRuntime 预测样例
python infer_resnet.py --model_file=./resnet50/inference.pdmodel --params_file=./resnet50/inference.pdiparams --use_onnxruntime=1
count_error


echo "demo 2: cpu_yolov3:"
# demo 2: cpu-yolov3 多输入模型 oneDNN/OnnxRuntime 预测样例
demo=`expr ${demo} + 1`

cd ../yolov3

if [ ! -d yolov3_r50vd_dcn_270e_coco ]; then
    wget -q https://paddle-inference-dist.bj.bcebos.com/Paddle-Inference-Demo/yolov3_r50vd_dcn_270e_coco.tgz
    tar xzf yolov3_r50vd_dcn_270e_coco.tgz
fi

if [ ! -f kite.jpg ]; then
    wget -q https://paddle-inference-dist.bj.bcebos.com/inference_demo/images/kite.jpg
fi

echo "demo 2.1 yolov3_mkldnn:"
# 使用 oneDNN 运行样例
python infer_yolov3.py --model_file=./yolov3_r50vd_dcn_270e_coco/model.pdmodel --params_file=./yolov3_r50vd_dcn_270e_coco/model.pdiparams
count_error

echo "demo 2.2 yolov3_onnxruntime:"
# 使用 OnnxRuntime 预测样例 - error
python infer_yolov3.py --model_file=./yolov3_r50vd_dcn_270e_coco/model.pdmodel --params_file=./yolov3_r50vd_dcn_270e_coco/model.pdiparams --use_onnxruntime=1
count_error


echo "demo 3: advanced-share_external_data-resnet50:"
# demo 3: advanced-share_external_data 运行 share_external_data 运行案例
demo=`expr ${demo} + 1`

cd ../../advanced/share_external_data/

sed -i "s/is not/!=/" infer_share_external_data.py

if [ ! -f ILSVRC2012_val_00000247.jpeg ]; then
    wget -q https://paddle-inference-dist.bj.bcebos.com/inference_demo/python/resnet50/ILSVRC2012_val_00000247.jpeg
fi

python infer_share_external_data.py --model_file=../../cpu/resnet50/resnet50/inference.pdmodel --params_file=../../cpu/resnet50/resnet50/inference.pdiparams
count_error


echo "demo 4: advanced-multi_thread-resnet50:"
# demo 4: advanced-multi_thread: 运行多线程运行样例
demo=`expr ${demo} + 1`

cd ../multi_thread/

sed -i "s/is not/!=/" threads_demo.py

if [ ! -f ILSVRC2012_val_00000247.jpeg ]; then
    wget -q https://paddle-inference-dist.bj.bcebos.com/inference_demo/python/resnet50/ILSVRC2012_val_00000247.jpeg
fi

python threads_demo.py --model_file=../../cpu/resnet50/resnet50/inference.pdmodel --params_file=../../cpu/resnet50/resnet50/inference.pdiparams --thread_num=2
count_error


echo "demo 5: advanced-custom_operator-resnet50:"
# demo 5: advanced-custom_operator 自定义算子执行
demo=`expr ${demo} + 1`

cd ../custom_operator/
python train_and_infer.py
count_error


echo "demo 6: gpu_resnet50:"
# demo 6: gpu-resnet50 单输入模型 原生 gpu/gpu 混合精度推理/Trt_fp32/Trt_fp16/Trt_int8/Trt_dynamic_shape 预测样例
demo=`expr ${demo} + 1`

cd ../../gpu/resnet50/

sed -i "s/is not/!=/" infer_resnet.py

if [ ! -f ILSVRC2012_val_00000247.jpeg ]; then
    wget -q https://paddle-inference-dist.bj.bcebos.com/inference_demo/python/resnet50/ILSVRC2012_val_00000247.jpeg
fi

echo "demo 6.1: gpu_resnet50_native:"
# 使用原生 GPU 运行样例
python infer_resnet.py --model_file=../../cpu/resnet50/resnet50/inference.pdmodel --params_file=../../cpu/resnet50/resnet50/inference.pdiparams
count_error

echo "demo 6.2: gpu_resnet50_mix:"
# 使用 GPU 混合精度推理 运行样例
# develop 分支下 -error
python infer_resnet.py --model_file=../../cpu/resnet50/resnet50/inference.pdmodel --params_file=../../cpu/resnet50/resnet50/inference.pdiparams --run_mode=gpu_fp16
count_error

echo "demo 6.3: fp32_resnet50:"
# 使用 Trt Fp32 运行样例
python infer_resnet.py --model_file=../../cpu/resnet50/resnet50/inference.pdmodel --params_file=../../cpu/resnet50/resnet50/inference.pdiparams --run_mode=trt_fp32
count_error

echo "demo 6.4: fp16_resnet50:"
# 使用 Trt Fp16 运行样例
python infer_resnet.py --model_file=../../cpu/resnet50/resnet50/inference.pdmodel --params_file=../../cpu/resnet50/resnet50/inference.pdiparams --run_mode=trt_fp16
count_error

echo "demo 6.5.1: trt_int8_resnet50_generate:"
# 使用 Trt Int8 运行样例
python infer_resnet.py --model_file=../../cpu/resnet50/resnet50/inference.pdmodel --params_file=../../cpu/resnet50/resnet50/inference.pdiparams --run_mode=trt_int8
count_error

echo "demo 6.5.2: trt_int8_resnet50_generate:"
python infer_resnet.py --model_file=../../cpu/resnet50/resnet50/inference.pdmodel --params_file=../../cpu/resnet50/resnet50/inference.pdiparams --run_mode=trt_int8
count_error

echo "demo 6.6: Try dynamic shape_resnet50:"
# 使用 Try dynamic shape 运行样例 -error
python infer_resnet.py --model_file=../../cpu/resnet50/resnet50/inference.pdmodel --params_file=../../cpu/resnet50/resnet50/inference.pdiparams --run_mode=trt_fp32 --use_dynamic_shape=1
count_error


echo "demo 7: gpu_yolov3:"
# demo 7: gpu-yolov3 多输入模型 原生 gpu/gpu 混合精度推理/Trt_fp32/Trt_fp16/Trt_int8/Trt_dynamic_shape 预测样例
demo=`expr ${demo} + 1`

cd ../yolov3/

# download data
if [ ! -f kite.jpg ]; then
    wget -q https://paddle-inference-dist.bj.bcebos.com/inference_demo/images/kite.jpg
fi

echo "demo 7.1: gpu_yolov3_native:"
# 使用原生 GPU 运行样例
python infer_yolov3.py --model_file=../../cpu/yolov3/yolov3_r50vd_dcn_270e_coco/model.pdmodel --params_file=../../cpu/yolov3/yolov3_r50vd_dcn_270e_coco/model.pdiparams
count_error

echo "demo 7.2: fp32_yolov3:"
# 使用 Trt Fp32 运行样例
python infer_yolov3.py --model_file=../../cpu/yolov3/yolov3_r50vd_dcn_270e_coco/model.pdmodel --params_file=../../cpu/yolov3/yolov3_r50vd_dcn_270e_coco/model.pdiparams --run_mode=trt_fp32
count_error


echo "demo 7.3: fp16_yolov3:"
# 使用 Trt Fp16 运行样例
python infer_yolov3.py --model_file=../../cpu/yolov3/yolov3_r50vd_dcn_270e_coco/model.pdmodel --params_file=../../cpu/yolov3/yolov3_r50vd_dcn_270e_coco/model.pdiparams --run_mode=trt_fp16
count_error


echo "demo 7.4.1: trt_int8_yolov3_generate:"
# 使用 Trt Int8 运行样例
python infer_yolov3.py --model_file=../../cpu/yolov3/yolov3_r50vd_dcn_270e_coco/model.pdmodel --params_file=../../cpu/yolov3/yolov3_r50vd_dcn_270e_coco/model.pdiparams --run_mode=trt_int8
count_error

echo "demo 7.4.2: trt_int8_yolov3:"
python infer_yolov3.py --model_file=../../cpu/yolov3/yolov3_r50vd_dcn_270e_coco/model.pdmodel --params_file=../../cpu/yolov3/yolov3_r50vd_dcn_270e_coco/model.pdiparams --run_mode=trt_int8
count_error

echo "demo 7.5: trt_dynamic_shape_yolov3:"
# 使用 Try dynamic shape 运行样例 -error
python infer_yolov3.py --model_file=../../cpu/yolov3/yolov3_r50vd_dcn_270e_coco/model.pdmodel --params_file=../../cpu/yolov3/yolov3_r50vd_dcn_270e_coco/model.pdiparams --run_mode=trt_fp32 --use_dynamic_shape=1
count_error


echo "demo 8: gpu_tuned_dynamic_shape:"
# demo 8: gpu-tuned_dynamic_shape Trt 动态 shape 自动推导 预测样例 使用 Paddle-TRT TunedDynamicShape 能力
demo=`expr ${demo} + 1`

cd ../tuned_dynamic_shape/

sed -i "s/is not/!=/" infer_tune.py

# 首先需要针对业务数据进行离线 tune，来获取计算图中所有中间 tensor 的 shape 范围，并将其存储在 config 中配置的 shape_range_info.pbtxt 文件中

echo "demo 8.1: gpu_tune:"
python infer_tune.py --model_file=../../cpu/resnet50/resnet50/inference.pdmodel --params_file=../../cpu/resnet50/resnet50/inference.pdiparams --tune 1
count_error

echo "demo 8.2: gpu_tuned_dynamic_shape:"
# 有了离线 tune 得到的 shape 范围信息后，您可以使用该文件自动对所有的 trt 子图设置其输入的 shape 范围。
python infer_tune.py --model_file=../../cpu/resnet50/resnet50/inference.pdmodel --params_file=../../cpu/resnet50/resnet50/inference.pdiparams --use_gpu=1 --use_trt 1 --tuned_dynamic_shape 1
count_error


echo "demo 9: ELMo_LAC:"
# demo 9: 基于 ELMo 的 LAC 分词预测样例
demo=`expr ${demo} + 1`

cd ../../mixed/ELMo/

if [ ! -d elmo ]; then
    wget -q https://paddle-inference-dist.bj.bcebos.com/Paddle-Inference-Demo/elmo.tgz
    tar xzf elmo.tgz 
fi

if [ ! -d elmo_data ]; then
    wget -q https://paddle-inference-dist.bj.bcebos.com/inference_demo/python/elmo/elmo_data.tgz
    tar xzf elmo_data.tgz
fi

python infer.py
count_error


echo "demo 10: mask_detection:"
# demo 10: 口罩检测预测样例
demo=`expr ${demo} + 1`
cd ../mask_detection/

python cam_video.py
count_error


echo "demo 11: LSTM INT8:"
# demo 11: LSTM INT8 prediction example on X86 Linux
demo=`expr ${demo} + 1`

cd ../x86_lstm_demo/

# 1. download model FP32 model for post-training quantization
if [ ! -d lstm_fp32_model ]; then
    wget -q https://paddle-inference-dist.bj.bcebos.com/lstm/lstm_fp32_model.tar.gz
    tar xzf lstm_fp32_model.tar.gz
fi

# 2. download model quant-aware model
if [ ! -d lstm_quant ]; then
    wget -q https://paddle-inference-dist.bj.bcebos.com/lstm/lstm_quant.tar.gz
    tar xzf lstm_quant.tar.gz
fi

# 3. download data
if [ ! -f quant_lstm_input_data ]; then
    wget -q https://paddle-inference-dist.bj.bcebos.com/int8/unittest_model_data/quant_lstm_input_data.tar.gz
    tar xzf quant_lstm_input_data.tar.gz
fi

# 4. download save_quant_model file
if [ ! -f save_quant_model.py ]; then
    wget -q https://raw.githubusercontent.com/PaddlePaddle/Paddle/develop/python/paddle/fluid/contrib/slim/tests/save_quant_model.py
fi

sed -i "s/python3/python/g" run.sh
bash run.sh
count_error

echo "demo 11.1: LSTM_INT8_thread:"
# 1 thread
# run fp32 model
python model_test.py --model_path=./lstm_fp32_model --data_path=./quant_lstm_input_data --use_analysis=False --num_threads=1
count_error

echo "demo 11.2: LSTM_INT8_4_threads:"
# 4 thread
# run fp32 model
python model_test.py --model_path=./lstm_fp32_model --data_path=./quant_lstm_input_data --use_analysis=False --num_threads=4
count_error


echo "total demos: "${demo}
echo "total cases: "${case}
echo "total errors: "${error}

cat /workspace/Paddle-Yu/Paddle-Inference-Demo/test_result.txt

exit ${error}
