# CoDeNet

Command to run Config a:
```
python test.py ctdet --arch shufflenetv2 --exp_id pascal_shufflenetv2_256_new1_1 --dataset pascal --input_res 256 --resume --flip_test --gpu 0
```

Command to run Config b:
```
python test.py ctdet --arch shufflenetv2 --exp_id pascal_shufflenetv2_256_new3_1 --dataset pascal --input_res 256 --resume --flip_test --gpu 0 --maxpool
```

Command to run Config c:
```
python test.py ctdet --arch shufflenetv2 --exp_id pascal_shufflenetv2_512_new14_1_test --dataset pascal --input_res 512 --resume --flip_test --gpu 0
```

Command to run Config d:
```
python test.py ctdet --arch shufflenetv2 --exp_id pascal_shufflenetv2_512_new17_1 --dataset pascal --input_res 512 --resume --flip_test --gpu 0 --w2
```

Command to run Config e:
```
python test.py ctdet --arch shufflenetv2 --exp_id pascal_shufflenetv2_512_new15_1 --dataset pascal --input_res 512 --resume --flip_test --gpu 0 --w2 --maxpool
```
