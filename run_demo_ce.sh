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

# 获取 PaddleClas 模型
git clone https://github.com/PaddlePaddle/PaddleClas.git --depth=1
cd PaddleClas

python -m pip install -r requirements.txt -i https://mirror.baidu.com/pypi/simple --trusted-host mirror.b+aidu.com

mkdir pretrained
mkdir inference_model

# ResNet50_vd
wget --no-proxy -P -q pretrained/ https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/ResNet50_vd_pretrained.pdparams
python tools/export_model.py -c ppcls/configs/ImageNet/ResNet/ResNet50_vd.yaml -o Global.pretrained_model=./pretrained/ResNet50_vd_pretrained -o Global.save_inference_dir=./inference_model/ResNet50_vd

# GhostNet_x1_0
wget --no-proxy -P -q pretrained/ https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/GhostNet_x1_0_pretrained.pdparams
python tools/export_model.py -c ppcls/configs/ImageNet/GhostNet/GhostNet_x1_0.yaml -o Global.pretrained_model=./pretrained/GhostNet_x1_0_pretrained -o Global.save_inference_dir=./inference_model/GhostNet_x1_0

# DenseNet121
wget --no-proxy -P -q pretrained/ https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/DenseNet121_pretrained.pdparams
python tools/export_model.py -c ppcls/configs/ImageNet/DenseNet/DenseNet121.yaml -o Global.pretrained_model=./pretrained/DenseNet121_pretrained -o Global.save_inference_dir=./inference_model/DenseNet121

# HRNet_W32_C
wget --no-proxy -P -q pretrained/ https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/HRNet_W32_C_pretrained.pdparams
python tools/export_model.py -c ppcls/configs/ImageNet/HRNet/HRNet_W32_C.yaml -o Global.pretrained_model=./pretrained/HRNet_W32_C_pretrained -o Global.save_inference_dir=./inference_model/HRNet_W32_C

# InceptionV4
wget --no-proxy -P -q pretrained/ https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/InceptionV4_pretrained.pdparams
python tools/export_model.py -c ppcls/configs/ImageNet/Inception/InceptionV4.yaml -o Global.pretrained_model=./pretrained/InceptionV4_pretrained -o Global.save_inference_dir=./inference_model/InceptionV4

# ViT_base_patch16_224
wget --no-proxy -P -q pretrained/ https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/ViT_base_patch16_224_pretrained.pdparams
python tools/export_model.py -c ppcls/configs/ImageNet/VisionTransformer/ViT_base_patch16_224.yaml -o Global.pretrained_model=./pretrained/ViT_base_patch16_224_pretrained -o Global.save_inference_dir=./inference_model/ViT_base_patch16_224

# DeiT_base_patch16_224
wget --no-proxy -P -q pretrained/ https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/DeiT_base_patch16_224_pretrained.pdparams
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

python -m pip install -r requirements.txt -i https://mirror.baidu.com/pypi/simple --trusted-host mirror.b+aidu.com

mkdir pretrained
mkdir inference_model

# faster_rcnn_r50_fpn_1x_coco
wget --no-proxy -P -q pretrained/ https://paddledet.bj.bcebos.com/models/faster_rcnn_r50_fpn_1x_coco.pdparams
python tools/export_model.py -c configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.yml --output_dir=./inference_model/ -o weights=pretrained/faster_rcnn_r50_fpn_1x_coco.pdparams

# mask_rcnn_r50_vd_fpn_2x_coco
wget --no-proxy -P -q pretrained/ https://paddledet.bj.bcebos.com/models/mask_rcnn_r50_vd_fpn_2x_coco.pdparams
python tools/export_model.py -c configs/mask_rcnn/mask_rcnn_r50_vd_fpn_2x_coco.yml --output_dir=./inference_model/ -o weights=pretrained/mask_rcnn_r50_vd_fpn_2x_coco.pdparams

# yolov3_darknet53_270e_coco
wget --no-proxy -P -q pretrained/ https://paddledet.bj.bcebos.com/models/yolov3_darknet53_270e_coco.pdparams
python tools/export_model.py -c configs/yolov3/yolov3_darknet53_270e_coco.yml --output_dir=./inference_model/ -o weights=pretrained/yolov3_darknet53_270e_coco.pdparams

# ssd_mobilenet_v1_300_120e_voc
wget --no-proxy -P -q pretrained/ https://paddledet.bj.bcebos.com/models/ssd_mobilenet_v1_300_120e_voc.pdparams
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
cd ../multi_stream && sed -i "s/TensorRT-7.1.3.4/TensorRT-8.0.3.4/" compile.sh && sh run.sh


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
