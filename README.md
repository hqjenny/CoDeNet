# CoDeNet Artifacts 

## 1. Object-Detection Network 

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

### 1.2 Ablation Study Results in Table 1 
Please follow the instructions at [CoDeNet_Detectron2][https://github.com/DequanWang/CoDeNet].

## 2. Object-Detection Accelerator
Please ssh to the remote ultra96 board by running `ssh root@358c4111r1.qicp.vip -p 7890`.

### 2.1 Latency Results in Table 5 and Figure 8. 
Please see the instructions in the ipython notebook `CoDeNet.ipynb` on the remote ultra96 board.

### 2.2 HLS Accelerator Source Code
Please refer to cpp files and the system files under `./hls`. 
The precompiled FPGA image is under `./bitfile`.
