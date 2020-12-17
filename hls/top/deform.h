#include<iostream>
#include "hls_stream.h"
#include "ap_int.h"
#include "para.h"
#include "conv.h"
#include "dma.h"
using namespace std;

void top
(		ap_int<IN_W * PE_3>* fmap,  // input feature map
        ap_int<OUT_W * PE_0>* out,  // output feature map
        ap_int<W_W * PA_0>* k0_1,   // conv1x1 weights
        ap_int<W_W * PE_3>* k0_3,   // conv3x3 weights
        ap_int<S_W>*quant,          // quantize parameter
        ap_uint<8>* offsets,        // deformable offsets
        int D,                      // image size
        int IC,                     // input channel number
        int OC,                     // output channel number
        int batch,                  // batch size
        bool STRIDE_2,              // 1 if stride=2
        bool skip3,                 // skip 3x3
        bool skip1,                 // skip 1x1
        bool deform,                // deformable flag 1: deform; 0:origin 3x3
        bool relu1,                 // relu flag for 1x1; 1 if use relu
        bool relu3                  // relu flag for 3x3; 1 if use relu
);

