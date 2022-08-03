#!/bin/bash
set +x
set -e 
pwd

git clone https://github.com/PaddlePaddle/Paddle-Inference-Demo.git --depth=1

# c++ 部分
# 下载预测库
cd Paddle-Inference-Demo/c++/lib/

wget -q https://paddle-inference-lib.bj.bcebos.com/2.3.1/cxx_c/Linux/GPU/x86-64_gcc8.2_avx_mkl_cuda11.2_cudnn8.2.1_trt8.0.3.4/paddle_inference.tgz
tar xzf paddle_inference.tgz


# demo 5 gpu-tuned_dynamic_shape
cd ../gpu/tuned_dynamic_shape/ && sed -i "s/TensorRT-7.1.3.4/TensorRT-8.0.3.4/" compile.sh

# 获取 PaddleDetection 模型
git clone https://github.com/PaddlePaddle/PaddleDetection.git --depth=1
cd PaddleDetection

python -m pip install -r requirements.txt -i https://mirror.baidu.com/pypi/simple --trusted-host mirror.b+aidu.com

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

cp -r ./inference_model/ ../

cd -

sh compile.sh detect

# yolov3_darknet53_270e_coco
# 离线tune测试
./build/detect --model_file inference_model/yolov3_darknet53_270e_coco/model.pdmodel --params_file inference_model/yolov3_darknet53_270e_coco/model.pdiparams --hs="608" --ws="608" --tune
# 动态shape及序列化测试
./build/detect --model_file inference_model/yolov3_darknet53_270e_coco/model.pdmodel --params_file inference_model/yolov3_darknet53_270e_coco/model.pdiparams --hs="608" --ws="608" --no_seen_hs="416" --no_seen_ws="416" --tuned_dynamic_shape --serialize
# 动态shape及反序列化测试
./build/detect --model_file inference_model/yolov3_darknet53_270e_coco/model.pdmodel --params_file inference_model/yolov3_darknet53_270e_coco/model.pdiparams --hs="608" --ws="608" --no_seen_hs="416" --no_seen_ws="416" --tuned_dynamic_shape --serialize


# ssd_mobilenet_v1_300_120e_voc
# 离线tune测试
./build/detect --model_file inference_model/ssd_mobilenet_v1_300_120e_voc//model.pdmodel --params_file inference_model/ssd_mobilenet_v1_300_120e_voc//model.pdiparams --hs="300" --ws="300" --tune
# 动态shape及序列化测试（注意：目前需要config加配置：config->Exp_DisableTensorRtOPs({"elementwise_add"});）
./build/detect --model_file inference_model/ssd_mobilenet_v1_300_120e_voc//model.pdmodel --params_file inference_model/ssd_mobilenet_v1_300_120e_voc//model.pdiparams --hs="300" --ws="300" --no_seen_hs="416" --no_seen_ws="416" --tuned_dynamic_shape --serialize
# 动态shape及反序列化测试（注意：目前需要config加配置：config->Exp_DisableTensorRtOPs({"elementwise_add"});）
./build/detect --model_file inference_model/ssd_mobilenet_v1_300_120e_voc//model.pdmodel --params_file inference_model/ssd_mobilenet_v1_300_120e_voc//model.pdiparams --hs="300" --ws="300" --no_seen_hs="416" --no_seen_ws="416" --tuned_dynamic_shape --serialize

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
# 动态shape及序列化测试（注意：目前需要config加配置：config->Exp_DisableTensorRtOPs({"roi_align", "elementwise_add"});）
./build/detect --model_file inference_model/mask_rcnn_r50_vd_fpn_2x_coco/model.pdmodel --params_file inference_model/mask_rcnn_r50_vd_fpn_2x_coco/model.pdiparams --hs="608" --ws="608" --no_seen_hs="416" --no_seen_ws="416" --tuned_dynamic_shape --serialize
# 动态shape及反序列化测试（注意：目前需要config加配置：config->Exp_DisableTensorRtOPs({"roi_align", "elementwise_add"});）
./build/detect --model_file inference_model/mask_rcnn_r50_vd_fpn_2x_coco/model.pdmodel --params_file inference_model/mask_rcnn_r50_vd_fpn_2x_coco/model.pdiparams --hs="608" --ws="608" --no_seen_hs="416" --no_seen_ws="416" --tuned_dynamic_shape --serialize
