# TVM for running first 8-8bit Layer
## 0. CoDeNet 1st Layer Latency on TVM 
Since we run the first layer with 8-bit weights, we need to run it on the ARM CPU. 
We implement the first layer on TVM and collect the latency results. 
Each layer is run 10 times and we take the average latency to report. 

### config a
```
Conv2DWorkload(batch=1, height=256, width=256, in_filter=3, out_filter=24, hkernel=3, wkernel=3, hpad=1, wpad=1, hstride=4, wstride=4)
CPU CONV2D TEST PASSED: Time cost = 0.0018344 sec/op, 2.89381 GOPS
```

**1st layer latency: 1.8ms**

### config b
```
Conv2DWorkload(batch=1, height=256, width=256, in_filter=3, out_filter=24, hkernel=3, wkernel=3, hpad=1, wpad=1, hstride=2, wstride=2)
CPU CONV2D TEST PASSED: Time cost = 0.00449328 sec/op, 4.72565 GOPS
Conv2DWorkload(batch=1, height=128, width=128, in_filter=24, out_filter=24, hkernel=3, wkernel=3, hpad=1, wpad=1, hstride=2, wstride=2)
ProfileResult(mean=0.0033822274, results=(0.0033822274,))
CPU POOLING TEST PASSED: Time cost = 0.00338223 sec/op
```
**1st layer latency: 7.9ms**

### config c
```
Conv2DWorkload(batch=1, height=512, width=512, in_filter=3, out_filter=24, hkernel=3, wkernel=3, hpad=1, wpad=1, hstride=4, wstride=4)
CPU CONV2D TEST PASSED: Time cost = 0.00500589 sec/op, 4.24174 GOPS
```
**1st layer latency: 5.0ms**
 
### config d
```
Conv2DWorkload(batch=1, height=512, width=512, in_filter=3, out_filter=24, hkernel=3, wkernel=3, hpad=1, wpad=1, hstride=4, wstride=4)
CPU CONV2D TEST PASSED: Time cost = 0.00492278 sec/op, 4.31335 GOPS
```
**1st layer latency: 5.0ms** 

### config e
```
Conv2DWorkload(batch=1, height=512, width=512, in_filter=3, out_filter=24, hkernel=3, wkernel=3, hpad=1, wpad=1, hstride=2, wstride=2)
CPU CONV2D TEST PASSED: Time cost = 0.0178155 sec/op, 4.76746 GOPS
Conv2DWorkload(batch=1, height=256, width=256, in_filter=24, out_filter=24, hkernel=3, wkernel=3, hpad=1, wpad=1, hstride=2, wstride=2)
ProfileResult(mean=0.0125795996, results=(0.0125795996,))
CPU POOLING TEST PASSED: Time cost = 0.0125796 sec/op
```
**1st layer latency: 30.4ms**

This results can be produced by following the instructions in the sections below.  

## 1. Setup
### a. TVM Setup
Download TVM source code and compile it.  
```
git clone --recursive https://github.com/apache/tvm tvm
cd tvm && git checkout c7a16d892da52f931a259a406922238a5e3e4f96 # this is the specific commit we use
```
Then follow the instructions on the [TVM installation page](https://tvm.apache.org/docs/install/from_source.html)
and the instructions on the [VTA installation page](https://tvm.apache.org/docs/vta/install.html#bitstream-generation-with-intel-toolchains)
to set up TVM.  

### b. Host Setup
On your host machine, first export the `TVM_PATH`:
```
export TVM_PATH=<path_to_your_tvm_repo>
```     
Then source the environment setup script:
```
source env.sh
```
Copy the VTA config file from this repo to the corresponding TVM path:
```
cp vta_config.json $TVM_PATH/3rdparty/vta-hw/config
``` 

### c. FPGA Setup

## 2. Start FPGA Server
Connect to the Ultra96 board via: 
```
ssh xilinx@192.168.2.1
```
The password is `xilinx`. Then start the TVM server from the remote FPGA with the following command: 
```
cd tvm && sudo ./apps/vta_rpc/start_rpc_server.sh # pw is 'xilinx'
```

## 3. Run the Host
Make sure the env variable `TVM_PATH` is set correctly and then 
copy all test files to the TVM repo and run them with `run_configs.sh`: 
```
cp run_configs.sh test_config* $TVM_PATH 
cd $TVM_PATH && ./run_configs.sh
```
