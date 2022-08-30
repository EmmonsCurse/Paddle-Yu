#!/bin/bash
set +x
set -e 
pwd

git clone https://github.com/PaddlePaddle/Paddle-Inference-Demo.git --depth=1
cd Paddle-Inference-Demo/python/cpu/resnet50


echo "Python Inference demo:"

# demo 1: cpu-resnet50 单输入模型 oneDNN/OnnxRuntime 预测样例
if [ ! -d resnet50 ]; then
    wget -q https://paddle-inference-dist.bj.bcebos.com/Paddle-Inference-Demo/resnet50.tgz
    tar xzf resnet50.tgz 
fi

if [ ! -f ILSVRC2012_val_00000247.jpeg ]; then
    wget -q https://paddle-inference-dist.bj.bcebos.com/inference_demo/python/resnet50/ILSVRC2012_val_00000247.jpeg
fi

# 使用 oneDNN 运行样例
python infer_resnet.py --model_file=./resnet50/inference.pdmodel --params_file=./resnet50/inference.pdiparams

# 使用 OnnxRuntime 预测样例
python infer_resnet.py --model_file=./resnet50/inference.pdmodel --params_file=./resnet50/inference.pdiparams --use_onnxruntime=1


# demo 2: cpu-yolov3 多输入模型 oneDNN/OnnxRuntime 预测样例
cd ../yolov3

if [ ! -d yolov3_r50vd_dcn_270e_coco ]; then
    wget -q https://paddle-inference-dist.bj.bcebos.com/Paddle-Inference-Demo/yolov3_r50vd_dcn_270e_coco.tgz
    tar xzf yolov3_r50vd_dcn_270e_coco.tgz
fi

if [ ! -f kite.jpg ]; then
    wget -q https://paddle-inference-dist.bj.bcebos.com/inference_demo/images/kite.jpg
fi

# 使用 oneDNN 运行样例
python infer_yolov3.py --model_file=./yolov3_r50vd_dcn_270e_coco/model.pdmodel --params_file=./yolov3_r50vd_dcn_270e_coco/model.pdiparams

# 使用 OnnxRuntime 预测样例
# python infer_yolov3.py --model_file=./yolov3_r50vd_dcn_270e_coco/model.pdmodel --params_file=./yolov3_r50vd_dcn_270e_coco/model.pdiparams --use_onnxruntime=1


# demo 3: advanced-share_external_data 运行 share_external_data 运行案例
cd ../../advanced/share_external_data/

if [ ! -f ILSVRC2012_val_00000247.jpeg ]; then
    wget -q https://paddle-inference-dist.bj.bcebos.com/inference_demo/python/resnet50/ILSVRC2012_val_00000247.jpeg
fi

python infer_share_external_data.py --model_file=../../cpu/resnet50/resnet50/inference.pdmodel --params_file=../../cpu/resnet50/resnet50/inference.pdiparams


# demo 4: advanced-multi_thread: 运行多线程运行样例
cd ../multi_thread/

if [ ! -f ILSVRC2012_val_00000247.jpeg ]; then
    wget -q https://paddle-inference-dist.bj.bcebos.com/inference_demo/python/resnet50/ILSVRC2012_val_00000247.jpeg
fi

python threads_demo.py --model_file=../../cpu/resnet50/resnet50/inference.pdmodel --params_file=../../cpu/resnet50/resnet50/inference.pdiparams --thread_num=2


# demo 5: advanced-custom_operator 自定义算子执行
cd ../custom_operator/
python train_and_infer.py


# demo 6: gpu-resnet50 单输入模型 原生gpu/gpu混合精度推理/Trt_fp32/Trt_fp16/Trt_int8/Trt_dynamic_shape 预测样例

cd ../../gpu/resnet50/

if [ ! -f ILSVRC2012_val_00000247.jpeg ]; then
    wget -q https://paddle-inference-dist.bj.bcebos.com/inference_demo/python/resnet50/ILSVRC2012_val_00000247.jpeg
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
    wget -q https://paddle-inference-dist.bj.bcebos.com/inference_demo/images/kite.jpg
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
    wget -q https://paddle-inference-dist.bj.bcebos.com/Paddle-Inference-Demo/elmo.tgz
    tar xzf elmo.tgz 
fi

if [ ! -d elmo_data ]; then
    wget -q https://paddle-inference-dist.bj.bcebos.com/inference_demo/python/elmo/elmo_data.tgz
    tar xzf elmo_data.tgz
fi

python infer.py


# demo 10: 口罩检测预测样例
cd ../mask_detection/


# demo 11: LSTM INT8 prediction example on X86 Linux
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



echo "C++ Inference demo:"

# 下载预测库
cd ../../../c++/lib/

wget -q https://paddle-inference-lib.bj.bcebos.com/2.3.1/cxx_c/Linux/GPU/x86-64_gcc8.2_avx_mkl_cuda11.2_cudnn8.2.1_trt8.0.3.4/paddle_inference.tgz
tar xzf paddle_inference.tgz

# demo 1 cpu-resnet50
cd ../cpu/resnet50/

sed -i "s/WITH_ONNXRUNTIME=ON/WITH_ONNXRUNTIME=OFF/" compile.sh

# 使用 oneDNN 运行样例
sh run.sh

# 使用 OnnxRuntime 运行样例
./build/resnet50_test --model_file resnet50/inference.pdmodel --params_file resnet50/inference.pdiparams --use_ort=1


# demo 2 cpu-yolov3
# 使用 oneDNN 运行样例
cd ../yolov3/ && sh run.sh

# 使用 OnnxRuntime 运行样例 -error
# ./build/yolov3_test --model_file yolov3_r50vd_dcn_270e_coco/model.pdmodel --params_file yolov3_r50vd_dcn_270e_coco/model.pdiparams --use_ort=1


# demo 3 gpu-resnet50
# 使用原生 GPU 运行样例
cd ../../gpu/resnet50/
sed -i "s/TensorRT-7.1.3.4/TensorRT-8.0.3.4/" compile.sh && sh run.sh

# 使用 TensorRT Fp32 运行样例
./build/resnet50_test --model_file resnet50/inference.pdmodel --params_file resnet50/inference.pdiparams --run_mode=trt_fp32

# 使用 TensorRT Fp16 运行样例
./build/resnet50_test --model_file resnet50/inference.pdmodel --params_file resnet50/inference.pdiparams --run_mode=trt_fp16


# 使用 TensorRT Int8 离线量化预测运行样例
# 生成量化校准表
./build/resnet50_test --model_file resnet50/inference.pdmodel --params_file resnet50/inference.pdiparams --run_mode=trt_int8

# 加载校准表执行预测
./build/resnet50_test --model_file resnet50/inference.pdmodel --params_file resnet50/inference.pdiparams --run_mode=trt_int8 --use_calib=true

# 使用 TensorRT 加载 PaddleSlim Int8 量化模型预测
# 模型下载 
wget -q https://paddle-inference-dist.bj.bcebos.com/inference_demo/python/resnet50/ResNet50_quant.tar.gz
tar -zxvf ResNet50_quant.tar.gz

./build/resnet50_test --model_dir ResNet50_quant/ --run_mode=trt_int8 --use_calib=false

# 使用 TensorRT dynamic shape 运行样例（以 Fp32 为例）-error
# ./build/resnet50_test --model_file resnet50/inference.pdmodel --params_file resnet50/inference.pdiparams --run_mode=trt_fp32 --use_dynamic_shape=1


# demo 4 gpu-yolov3
# 使用原生 GPU 运行样例
cd ../yolov3/ && sed -i "s/TensorRT-7.1.3.4/TensorRT-8.0.3.4/" compile.sh && sh run.sh

# 使用 TensorRT Fp32 运行样例
./build/yolov3_test --model_file yolov3_r50vd_dcn_270e_coco/model.pdmodel --params_file yolov3_r50vd_dcn_270e_coco/model.pdiparams --run_mode=trt_fp32

# 使用 TensorRT Fp16 运行样例
./build/yolov3_test --model_file yolov3_r50vd_dcn_270e_coco/model.pdmodel --params_file yolov3_r50vd_dcn_270e_coco/model.pdiparams --run_mode=trt_fp16

# 使用 TensorRT Int8 运行样例
# 生成量化校准表
./build/yolov3_test --model_file yolov3_r50vd_dcn_270e_coco/model.pdmodel --params_file yolov3_r50vd_dcn_270e_coco/model.pdiparams --run_mode=trt_int8

# 加载校准表执行预测
./build/yolov3_test --model_file yolov3_r50vd_dcn_270e_coco/model.pdmodel --params_file yolov3_r50vd_dcn_270e_coco/model.pdiparams --run_mode=trt_int8

# 使用 TensorRT dynamic shape 运行样例（以 Fp32 为例）-error
# ./build/yolov3_test --model_file yolov3_r50vd_dcn_270e_coco/inference.pdmodel --params_file yolov3_r50vd_dcn_270e_coco/inference.pdiparams --run_mode=trt_fp32 --use_dynamic_shape=1


# demo 5 gpu-tuned_dynamic_shape
cd ../tuned_dynamic_shape/ && sed -i "s/TensorRT-7.1.3.4/TensorRT-8.0.3.4/" compile.sh

# 获取 PaddleClas 模型
git clone https://github.com/PaddlePaddle/PaddleClas.git --depth=1
cd PaddleClas

# 安装环境依赖
python setup.py build
python setup.py install

mkdir pretrained
mkdir inference_model

# ResNet50_vd
wget --no-proxy -P pretrained/ https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/ResNet50_vd_pretrained.pdparams
python tools/export_model.py -c ppcls/configs/ImageNet/ResNet/ResNet50_vd.yaml -o Global.pretrained_model=./pretrained/ResNet50_vd_pretrained -o Global.save_inference_dir=./inference_model/ResNet50_vd

# GhostNet_x1_0
wget --no-proxy -P pretrained/ https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/GhostNet_x1_0_pretrained.pdparams
python tools/export_model.py -c ppcls/configs/ImageNet/GhostNet/GhostNet_x1_0.yaml -o Global.pretrained_model=./pretrained/GhostNet_x1_0_pretrained -o Global.save_inference_dir=./inference_model/GhostNet_x1_0

# DenseNet121
wget --no-proxy -P pretrained/ https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/DenseNet121_pretrained.pdparams
python tools/export_model.py -c ppcls/configs/ImageNet/DenseNet/DenseNet121.yaml -o Global.pretrained_model=./pretrained/DenseNet121_pretrained -o Global.save_inference_dir=./inference_model/DenseNet121

# HRNet_W32_C
wget --no-proxy -P pretrained/ https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/HRNet_W32_C_pretrained.pdparams
python tools/export_model.py -c ppcls/configs/ImageNet/HRNet/HRNet_W32_C.yaml -o Global.pretrained_model=./pretrained/HRNet_W32_C_pretrained -o Global.save_inference_dir=./inference_model/HRNet_W32_C

# InceptionV4
wget --no-proxy -P pretrained/ https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/InceptionV4_pretrained.pdparams
python tools/export_model.py -c ppcls/configs/ImageNet/Inception/InceptionV4.yaml -o Global.pretrained_model=./pretrained/InceptionV4_pretrained -o Global.save_inference_dir=./inference_model/InceptionV4

# ViT_base_patch16_224
wget --no-proxy -P pretrained/ https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/ViT_base_patch16_224_pretrained.pdparams
python tools/export_model.py -c ppcls/configs/ImageNet/VisionTransformer/ViT_base_patch16_224.yaml -o Global.pretrained_model=./pretrained/ViT_base_patch16_224_pretrained -o Global.save_inference_dir=./inference_model/ViT_base_patch16_224

# DeiT_base_patch16_224
wget --no-proxy -P pretrained/ https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/DeiT_base_patch16_224_pretrained.pdparams
python tools/export_model.py -c ppcls/configs/ImageNet/DeiT/DeiT_base_patch16_224.yaml -o Global.pretrained_model=./pretrained/DeiT_base_patch16_224_pretrained -o Global.save_inference_dir=./inference_model/DeiT_base_patch16_224

# 复制推理模型
cp -r ./inference_model/ ../
cd -

# TunedDynamicShape 测试
sh compile.sh clas

# 注意以下CV类的测试case，不是真正的变长模型，对输入shape是有要求的，无法支持任意的输入shape

# ResNet50_Vd
# 离线tune测试
./build/clas --model_file inference_model/ResNet50_vd/inference.pdmodel  --params_file inference_model/ResNet50_vd/inference.pdiparams --hs="224:448" --ws="224:448" --tune
# 动态shape及序列化测试
./build/clas --model_file inference_model/ResNet50_vd/inference.pdmodel  --params_file inference_model/ResNet50_vd/inference.pdiparams --hs="224:448" --ws="224:448" --no_seen_hs="112:672" --no_seen_ws="112:672" --tuned_dynamic_shape --serialize
# 动态shape及反序列化测试
./build/clas --model_file inference_model/ResNet50_vd/inference.pdmodel  --params_file inference_model/ResNet50_vd/inference.pdiparams --hs="224:448" --ws="224:448" --no_seen_hs="112:672" --no_seen_ws="112:672" --tuned_dynamic_shape --serialize

# GhostNet_x1_0
# 离线tune测试
./build/clas --model_file inference_model/GhostNet_x1_0/inference.pdmodel --params_file inference_model/GhostNet_x1_0/inference.pdiparams --hs="224:448" --ws="224:448" --tune
# 动态shape及序列化测试
./build/clas --model_file inference_model/GhostNet_x1_0/inference.pdmodel --params_file inference_model/GhostNet_x1_0/inference.pdiparams --hs="224:448" --ws="224:448" --no_seen_hs="112:672" --no_seen_ws="112:672" --tuned_dynamic_shape --serialize
# 动态shape及反序列化测试
./build/clas --model_file inference_model/GhostNet_x1_0/inference.pdmodel --params_file inference_model/GhostNet_x1_0/inference.pdiparams --hs="224:448" --ws="224:448" --no_seen_hs="112:672" --no_seen_ws="112:672" --tuned_dynamic_shape --serialize

# DenseNet121
# 离线tune测试
./build/clas --model_file inference_model/DenseNet121/inference.pdmodel --params_file inference_model/DenseNet121/inference.pdiparams --hs="224:448" --ws="224:448" --tune
# 动态shape及序列化测试
./build/clas --model_file inference_model/DenseNet121/inference.pdmodel --params_file inference_model/DenseNet121/inference.pdiparams --hs="224:448" --ws="224:448" --no_seen_hs="112:672" --no_seen_ws="112:672" --tuned_dynamic_shape --serialize
# 动态shape及反序列化测试
./build/clas --model_file inference_model/DenseNet121/inference.pdmodel --params_file inference_model/DenseNet121/inference.pdiparams --hs="224:448" --ws="224:448" --no_seen_hs="112:672" --no_seen_ws="112:672" --tuned_dynamic_shape --serialize

# HRNet_W32_C
# 离线tune测试
./build/clas --model_file inference_model/HRNet_W32_C/inference.pdmodel --params_file inference_model/HRNet_W32_C/inference.pdiparams --hs="224" --ws="224" --tune
# 动态shape及序列化测试
./build/clas --model_file inference_model/HRNet_W32_C/inference.pdmodel --params_file inference_model/HRNet_W32_C/inference.pdiparams --hs="224" --ws="224" --no_seen_hs="448" --no_seen_ws="448" --tuned_dynamic_shape --serialize
# 动态shape及反序列化测试
./build/clas --model_file inference_model/HRNet_W32_C/inference.pdmodel --params_file inference_model/HRNet_W32_C/inference.pdiparams --hs="224" --ws="224" --no_seen_hs="448" --no_seen_ws="448" --tuned_dynamic_shape --serialize

# InceptionV4
# 离线tune测试
./build/clas --model_file inference_model/InceptionV4/inference.pdmodel --params_file inference_model/InceptionV4/inference.pdiparams --hs="224" --ws="224" --tune
# 动态shape及序列化测试
./build/clas --model_file inference_model/InceptionV4/inference.pdmodel --params_file inference_model/InceptionV4/inference.pdiparams --hs="224" --ws="224" --no_seen_hs="448" --no_seen_ws="448" --tuned_dynamic_shape --serialize
# 动态shape及反序列化测试
./build/clas --model_file inference_model/InceptionV4/inference.pdmodel --params_file inference_model/InceptionV4/inference.pdiparams --hs="224" --ws="224" --no_seen_hs="448" --no_seen_ws="448" --tuned_dynamic_shape --serialize

# ViT_base_patch16_224

# config 加配置
sed -i "74s#// config#config#g" clas.cc
sh compile.sh clas

# 离线tune测试
./build/clas --model_file inference_model/ViT_base_patch16_224/inference.pdmodel --params_file inference_model/ViT_base_patch16_224/inference.pdiparams --hs="224" --ws="224" --tune
# 动态shape及序列化测试（注意：目前需要config加配置：config->Exp_DisableTensorRtOPs({"elementwise_add"});）
./build/clas --model_file inference_model/ViT_base_patch16_224/inference.pdmodel --params_file inference_model/ViT_base_patch16_224/inference.pdiparams --hs="224" --ws="224"--no_seen_hs="224" --no_seen_ws="224" --tuned_dynamic_shape --serialize
# 动态shape及反序列化测试（注意：目前需要config加配置：config->Exp_DisableTensorRtOPs({"elementwise_add"});）
./build/clas --model_file inference_model/ViT_base_patch16_224/inference.pdmodel --params_file inference_model/ViT_base_patch16_224/inference.pdiparams --hs="224" --ws="224"--no_seen_hs="224" --no_seen_ws="224" --tuned_dynamic_shape --serialize

# DeiT_base_patch16_224
# 离线tune测试
./build/clas --model_file inference_model/DeiT_base_patch16_224/inference.pdmodel --params_file inference_model/DeiT_base_patch16_224/inference.pdiparams --hs="224" --ws="224" --tune
# 动态shape及序列化测试（注意：目前需要config加配置：config->Exp_DisableTensorRtOPs({"elementwise_add"});）
./build/clas --model_file inference_model/DeiT_base_patch16_224/inference.pdmodel --params_file inference_model/DeiT_base_patch16_224/inference.pdiparams --hs="224" --ws="224"--no_seen_hs="224" --no_seen_ws="224" --tuned_dynamic_shape --serialize
# 动态shape及反序列化测试（注意：目前需要config加配置：config->Exp_DisableTensorRtOPs({"elementwise_add"});）
./build/clas --model_file inference_model/DeiT_base_patch16_224/inference.pdmodel --params_file inference_model/DeiT_base_patch16_224/inference.pdiparams --hs="224" --ws="224"--no_seen_hs="224" --no_seen_ws="224" --tuned_dynamic_shape --serialize


# 获取 PaddleDetection 模型
git clone https://github.com/PaddlePaddle/PaddleDetection.git --depth=1
cd PaddleDetection

# 安装环境依赖
python setup.py build
python setup.py install
# python -m pip install -r requirements.txt -i https://mirror.baidu.com/pypi/simple --trusted-host mirror.b+aidu.com

mkdir pretrained
mkdir inference_model

# faster_rcnn_r50_fpn_1x_coco
wget --no-proxy -P pretrained/ https://paddledet.bj.bcebos.com/models/faster_rcnn_r50_fpn_1x_coco.pdparams
python tools/export_model.py -c configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.yml --output_dir=./inference_model/ -o weights=pretrained/faster_rcnn_r50_fpn_1x_coco.pdparams

# mask_rcnn_r50_vd_fpn_2x_coco
wget --no-proxy -P pretrained/ https://paddledet.bj.bcebos.com/models/mask_rcnn_r50_vd_fpn_2x_coco.pdparams
python tools/export_model.py -c configs/mask_rcnn/mask_rcnn_r50_vd_fpn_2x_coco.yml --output_dir=./inference_model/ -o weights=pretrained/mask_rcnn_r50_vd_fpn_2x_coco.pdparams

# yolov3_darknet53_270e_coco
wget --no-proxy -P pretrained/ https://paddledet.bj.bcebos.com/models/yolov3_darknet53_270e_coco.pdparams
python tools/export_model.py -c configs/yolov3/yolov3_darknet53_270e_coco.yml --output_dir=./inference_model/ -o weights=pretrained/yolov3_darknet53_270e_coco.pdparams

# ssd_mobilenet_v1_300_120e_voc
wget --no-proxy -P pretrained/ https://paddledet.bj.bcebos.com/models/ssd_mobilenet_v1_300_120e_voc.pdparams
python tools/export_model.py -c configs/ssd/ssd_mobilenet_v1_300_120e_voc.yml --output_dir=./inference_model/ -o weights=pretrained/ssd_mobilenet_v1_300_120e_voc.pdparams

cp -r ./inference_model/* ../inference_model/
cd -


sh compile.sh detect

# yolov3_darknet53_270e_coco
# 离线tune测试
./build/detect --model_file inference_model/yolov3_darknet53_270e_coco/model.pdmodel --params_file inference_model/yolov3_darknet53_270e_coco/model.pdiparams --hs="608" --ws="608" --tune
# 动态shape及序列化测试
./build/detect --model_file inference_model/yolov3_darknet53_270e_coco/model.pdmodel --params_file inference_model/yolov3_darknet53_270e_coco/model.pdiparams --hs="608" --ws="608" --no_seen_hs="416" --no_seen_ws="416" --tuned_dynamic_shape --serialize
# 动态shape及反序列化测试
./build/detect --model_file inference_model/yolov3_darknet53_270e_coco/model.pdmodel --params_file inference_model/yolov3_darknet53_270e_coco/model.pdiparams --hs="608" --ws="608" --no_seen_hs="416" --no_seen_ws="416" --tuned_dynamic_shape --serialize

# config 修改
sed -i "75s#// config#config#g" detect.cc
sh compile.sh detect

# ssd_mobilenet_v1_300_120e_voc
# 离线tune测试
./build/detect --model_file inference_model/ssd_mobilenet_v1_300_120e_voc//model.pdmodel --params_file inference_model/ssd_mobilenet_v1_300_120e_voc//model.pdiparams --hs="300" --ws="300" --tune
# 动态shape及序列化测试（注意：目前需要config加配置：config->Exp_DisableTensorRtOPs({"elementwise_add"});）
./build/detect --model_file inference_model/ssd_mobilenet_v1_300_120e_voc//model.pdmodel --params_file inference_model/ssd_mobilenet_v1_300_120e_voc//model.pdiparams --hs="300" --ws="300" --no_seen_hs="416" --no_seen_ws="416" --tuned_dynamic_shape --serialize
# 动态shape及反序列化测试（注意：目前需要config加配置：config->Exp_DisableTensorRtOPs({"elementwise_add"});）
./build/detect --model_file inference_model/ssd_mobilenet_v1_300_120e_voc//model.pdmodel --params_file inference_model/ssd_mobilenet_v1_300_120e_voc//model.pdiparams --hs="300" --ws="300" --no_seen_hs="416" --no_seen_ws="416" --tuned_dynamic_shape --serialize


# 修改
# sed -i "s#"roi_align"#"roi_align\",\"elementwise_add"#" detect.cc

# faster_rcnn_r50_fpn_1x_coco
# 离线tune测试
./build/detect --model_file inference_model/faster_rcnn_r50_fpn_1x_coco/model.pdmodel --params_file inference_model/faster_rcnn_r50_fpn_1x_coco/model.pdiparams --hs="608" --ws="608" --tune
# 动态shape及序列化测试（注意：目前需要config加配置：config->Exp_DisableTensorRtOPs({"roi_align", "elementwise_add"});）
./build/detect --model_file inference_model/faster_rcnn_r50_fpn_1x_coco/model.pdmodel --params_file inference_model/faster_rcnn_r50_fpn_1x_coco/model.pdiparams --hs="608" --ws="608" --no_seen_hs="416" --no_seen_ws="416" --tuned_dynamic_shape --serialize
# 动态shape及反序列化测试（注意：目前需要config加配置：config->Exp_DisableTensorRtOPs({"roi_align", "elementwise_add"});）
./build/detect --model_file inference_model/faster_rcnn_r50_fpn_1x_coco/model.pdmodel --params_file inference_model/faster_rcnn_r50_fpn_1x_coco/model.pdiparams --hs="608" --ws="608" --no_seen_hs="416" --no_seen_ws="416" --tuned_dynamic_shape --serialize

# mask_rcnn_r50_vd_fpn_2x_coco
# 离线tune测试
./build/detect --model_file inference_model/mask_rcnn_r50_vd_fpn_2x_coco/model.pdmodel --params_file inference_model/mask_rcnn_r50_vd_fpn_2x_coco/model.pdiparams --hs="608" --ws="608" --tune

# # 动态shape及序列化测试（注意：目前需要config加配置：config->Exp_DisableTensorRtOPs({"roi_align", "elementwise_add"});）
# ./build/detect --model_file inference_model/mask_rcnn_r50_vd_fpn_2x_coco/model.pdmodel --params_file inference_model/mask_rcnn_r50_vd_fpn_2x_coco/model.pdiparams --hs="608" --ws="608" --no_seen_hs="416" --no_seen_ws="416" --tuned_dynamic_shape --serialize
# # 动态shape及反序列化测试（注意：目前需要config加配置：config->Exp_DisableTensorRtOPs({"roi_align", "elementwise_add"});）
# ./build/detect --model_file inference_model/mask_rcnn_r50_vd_fpn_2x_coco/model.pdmodel --params_file inference_model/mask_rcnn_r50_vd_fpn_2x_coco/model.pdiparams --hs="608" --ws="608" --no_seen_hs="416" --no_seen_ws="416" --tuned_dynamic_shape --serialize


# PaddleOCR 模型测试
mkdir ocr_inference_model && cd ocr_inference_model

wget -q https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_det_infer.tar
tar xf ch_ppocr_mobile_v2.0_det_infer.tar

wget -q https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_cls_infer.tar
tar xf ch_ppocr_mobile_v2.0_cls_infer.tar

wget -q https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_rec_infer.tar
tar xf ch_ppocr_mobile_v2.0_rec_infer.tar

wget -q https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_server_v2.0_det_infer.tar
tar xf ch_ppocr_server_v2.0_det_infer.tar

wget -q https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_server_v2.0_rec_infer.tar
tar xf ch_ppocr_server_v2.0_rec_infer.tar

cd -

# ocr det模型测试
sh compile.sh ocr_det

# 离线tune测试
./build/ocr_det --model_file ocr_inference_model/ch_ppocr_server_v2.0_det_infer/inference.pdmodel --params_file ocr_inference_model/ch_ppocr_server_v2.0_det_infer/inference.pdiparams --hs="640" --ws="640" --tune
# 动态shape及序列化测试
./build/ocr_det --model_file ocr_inference_model/ch_ppocr_server_v2.0_det_infer/inference.pdmodel --params_file ocr_inference_model/ch_ppocr_server_v2.0_det_infer/inference.pdiparams --hs="640" --ws="640" --no_seen_hs="320" --no_seen_ws="320" --tuned_dynamic_shape --serialize
# 动态shape及反序列化测试
./build/ocr_det --model_file ocr_inference_model/ch_ppocr_server_v2.0_det_infer/inference.pdmodel --params_file ocr_inference_model/ch_ppocr_server_v2.0_det_infer/inference.pdiparams --hs="640" --ws="640" --no_seen_hs="320" --no_seen_ws="320" --tuned_dynamic_shape --serialize

# ocr cls模型测试
sh compile.sh ocr_cls

# 离线tune测试
./build/ocr_cls --model_file ocr_inference_model/ch_ppocr_mobile_v2.0_cls_infer/inference.pdmodel --params_file ocr_inference_model/ch_ppocr_mobile_v2.0_cls_infer/inference.pdiparams --hs="640" --ws="640" --tune
# 动态shape及序列化测试
./build/ocr_cls --model_file ocr_inference_model/ch_ppocr_mobile_v2.0_cls_infer/inference.pdmodel --params_file ocr_inference_model/ch_ppocr_mobile_v2.0_cls_infer/inference.pdiparams --hs="640" --ws="640" --no_seen_hs="320" --no_seen_ws="320" --tuned_dynamic_shape --serialize
# 动态shape及反序列化测试
./build/ocr_cls --model_file ocr_inference_model/ch_ppocr_mobile_v2.0_cls_infer/inference.pdmodel --params_file ocr_inference_model/ch_ppocr_mobile_v2.0_cls_infer/inference.pdiparams --hs="640" --ws="640" --no_seen_hs="320" --no_seen_ws="320" --tuned_dynamic_shape --serialize


# ocr rec模型测试
sh compile.sh ocr_rec

# 离线tune测试
./build/ocr_rec --model_file ocr_inference_model/ch_ppocr_server_v2.0_rec_infer/inference.pdmodel --params_file ocr_inference_model/ch_ppocr_server_v2.0_rec_infer/inference.pdiparams --hs="32" --ws="32" --tune
# 动态shape及序列化测试
./build/ocr_rec --model_file ocr_inference_model/ch_ppocr_server_v2.0_rec_infer/inference.pdmodel --params_file ocr_inference_model/ch_ppocr_server_v2.0_rec_infer/inference.pdiparams --hs="32" --ws="32" --no_seen_hs="32" --no_seen_ws="320" --tuned_dynamic_shape --serialize
# 动态shape及反序列化测试
./build/ocr_rec --model_file ocr_inference_model/ch_ppocr_server_v2.0_rec_infer/inference.pdmodel --params_file ocr_inference_model/ch_ppocr_server_v2.0_rec_infer/inference.pdiparams --hs="32" --ws="32" --no_seen_hs="32" --no_seen_ws="320" --tuned_dynamic_shape --serialize



# PaddleNLP 模型测试
mkdir nlp_inference_model && cd nlp_inference_model
wget -q http://paddle-inference-dist.bj.bcebos.com/tensorrt_test/ernie_model_4.tar.gz
tar xzf ernie_model_4.tar.gz
cd -

sh compile.sh ernie

# 离线tune测试
./build/ernie --model_dir nlp_inference_model/ernie_model_4 --tune
# 动态shape及序列化测试
./build/ernie --model_dir nlp_inference_model/ernie_model_4 --tuned_dynamic_shape --serialize
# 动态shape及反序列化测试
./build/ernie --model_dir nlp_inference_model/ernie_model_4 --tuned_dynamic_shape --serialize



# demo 6 gpu-ernie-varlen
cd ../ernie-varlen/ && sed -i "s#TENSORRT_ROOT=/root/work/nvidia/TensorRT-7.2.3.4.cuda-10.1.cudnn7.6-OSS7.2.1#TENSORRT_ROOT=/usr/local/TensorRT-8.0.3.4#" compile.sh
sh compile.sh

if [ ! -d ernie_model_4 ]; then
    wget -q http://paddle-inference-dist.bj.bcebos.com/tensorrt_test/ernie_model_4.tar.gz;
    tar xzf ernie_model_4.tar.gz;
fi

./build/ernie_varlen_test --model_dir=./ernie_model_4/


# demo 7 gpu-gpu_fp16
# 使用原生 GPU 运行样例
cd ../gpu_fp16 && sed -i "s/TensorRT-7.1.3.4/TensorRT-8.0.3.4/" compile.sh && sh run.sh

# 使用 GPU 混合精度推理 运行样例
./build/resnet50_gpu_fp16 --model_file resnet50/inference.pdmodel --params_file resnet50/inference.pdiparams --use_gpu_fp16=1


# demo 8 gpu-multi_stream
# 运行 GPU 多流预测样例 
# 该功能 develop 和 2.4 后可用
# cd ../multi_stream && sed -i "s/TensorRT-7.1.3.4/TensorRT-8.0.3.4/" compile.sh && sh run.sh


# demo 9 advanced-custom_operator
# 自定义算子模型构建运行示例
cd ../../advanced/custom-operator/ && sed -i "s/WITH_ONNXRUNTIME=ON/WITH_ONNXRUNTIME=OFF/" compile.sh && sed -i "s/TensorRT-7.1.3.4/TensorRT-8.0.3.4/" compile.sh && sh run.sh


# demo 10 advanced-share_external_data
# 运行 share_external_data 预测样例
# 运行 CPU 样例
cd ../share_external_data/ && sed -i "s/TensorRT-7.1.3.4/TensorRT-8.0.3.4/" compile.sh && sh run.sh

# 运行 GPU 样例
./build/resnet50_share_data --model_file resnet50/inference.pdmodel --params_file resnet50/inference.pdiparams --use_gpu=1


# demo 11 advanced-multi_thread
# 运行多线程预测样例
cd ../share_external_data/ && sed -i "s/TensorRT-7.1.3.4/TensorRT-8.0.3.4/" compile.sh && sh run.sh


# demo 12 advanced


# demo 13 mixed-LIC2020
# 运行C++ LIC2020关系抽取 demo
cd ../../mixed/LIC2020/

sed -i "s/TensorRT-7.1.3.4/TensorRT-8.0.3.4/" compile.sh

sed -i "s#${work_path}/../lib/#${work_path}/../../lib/#" compile.sh && sh run.sh
