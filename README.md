# CoDeNet: Efficient Deployment of Input-Adaptive Object Detection on Embedded FPGAs
[![DOI](https://zenodo.org/badge/318716328.svg)](https://zenodo.org/badge/latestdoi/318716328)

## 1. Object-Detection Networks 

The code of quantized object-detection network model for hardware acceleration is under [dnn/CoDeNet](https://github.com/Zhen-Dong/CoDeNet) submodule. We also have non-quantized models to study impact of hardware-friendly deformable convolution modifications. These models are compatiable with the [Detectron2](https://github.com/facebookresearch/detectron2) library and are under [dnn/CoDeNet_Detectron2](https://github.com/DequanWang/CoDeNet/tree/2e80791e743b14ba3cb8be4f2403921aa395c9aa). 

### 1.1 Experimental Results in Table 3. 
#### Quantized Network Accuracy Table 3. row 3  
Command to run Config a:
```
python test.py ctdet --arch shufflenetv2 --exp_id pascal_shufflenetv2_256_new1_1 --dataset pascal --input_res 256 --resume --flip_test --gpu 0
```

#### Quantized Network Accuracy Table 3. row 5 
Command to run Config b:
```
python test.py ctdet --arch shufflenetv2 --exp_id pascal_shufflenetv2_256_new3_1 --dataset pascal --input_res 256 --resume --flip_test --gpu 0 --maxpool
```

#### Quantized Network Accuracy Table 3. row 7 
Command to run Config c:
```
python test.py ctdet --arch shufflenetv2 --exp_id pascal_shufflenetv2_512_new14_1_test --dataset pascal --input_res 512 --resume --flip_test --gpu 0
```

#### Quantized Network Accuracy Table 3. row 9 
Command to run Config d:
```
python test.py ctdet --arch shufflenetv2 --exp_id pascal_shufflenetv2_512_new17_1 --dataset pascal --input_res 512 --resume --flip_test --gpu 0 --w2
```

#### Quantized Network Accuracy Table 3. row 11 
Command to run Config e:
```
python test.py ctdet --arch shufflenetv2 --exp_id pascal_shufflenetv2_512_new15_1 --dataset pascal --input_res 512 --resume --flip_test --gpu 0 --w2 --maxpool
```

### 1.2 Ablation Study Results in Table 1. 
Please follow the instructions at [CoDeNet_Detectron2 Installation](https://github.com/DequanWang/CoDeNet/blob/master/INSTALL.md) to set up the environment.
We also provide a remote server to evaluate the trained model. 

Command to run VOC result with modified deformable convolution in Table 1. last row: 
```
python tools/train_net.py --num-gpus 1 --config-file configs/centernet/voc/V2_1.0x_voc_512_4gpus_1x_deform_conv_square_depthwise.yaml --eval-only MODEL.WEIGHTS output/centernet/voc/V2_1.0x_voc_512_4gpus_1x_deform_conv_square_depthwise/model_final.pth 
# result: AP: 41.7	AP50: 64.5	AP75: 43.8
```
Command to run COCO result with modified deformable convolution in Table 1. last row: 
```
python tools/train_net.py --num-gpus 1 --config-file configs/centernet/coco/V2_1.0x_coco_512_10gpus_1x_deform_conv_square_depthwise.yaml --eval-only MODEL.WEIGHTS output/centernet/coco/V2_1.0x_coco_512_10gpus_1x_deform_conv_square_depthwise/model_final.pth 
# result: AP: 21.6	AP50: 37.4	AP75: 21.8	APs: 6.5	APm: 23.7	APl: 34.8
```

## 2. Object-Detection Accelerator
We evaluate the latency of our network on the [Ultra96 PYNQ platform](https://ultra96-pynq.readthedocs.io/en/latest/index.html). 

### 2.1 HLS Accelerator Source Code
Please refer to cpp files and the system files under [./hls](hls). 
The precompiled FPGA image is under [./bitfile](bitfile).
The project file can be downloaded [here](https://people.eecs.berkeley.edu/~qijing.huang/2021FPGA/CoDeNet.xpr.zip). 
The hls project can be downloaded [here](https://people.eecs.berkeley.edu/~qijing.huang/2021FPGA/CoDeNet_hls.zip).

### 2.2 Software Invocation Source Code 
The source code for running the first layer layer latency is under [sw/tvm](sw/tvm). Please follow the [sw/tvm/README.md](sw/tvm/README.md) to run it. 
The source code for calling the accelearator is in [codenet.ipynb](sw/codenet.ipynb). 

### 2.3 Latency Results in Table 5 and Figure 8. 
Please connect to the Ultra96 board and browse to the ipython notebook page `http://192.168.2.1:9090/`.
Upload the `sw/codenet.ipynb` and `sw/bitfile` folder to the remote FPGA. Run the iptyon notebook to see the latency results. 

