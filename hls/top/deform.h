#include<iostream>
#include "hls_stream.h"
#include "ap_int.h"
#include "para.h"
#include "conv.h"
#include "dma.h"
using namespace std;

void top
(		ap_int<IN_W * PE_3>* fmap,
		ap_int<OUT_W * PE_0>* out,
		ap_int<W_W * PA_0>* k0_1,
		ap_int<W_W * PE_3>* k0_3,
		ap_int<S_W>*quant,
		ap_uint<8>* dilation,
		int D,
		int IC,
		int OC,
		int batch,
		bool STRIDE_2,
		bool skip3,
		bool skip1,
		bool deform,
		bool relu1,
		bool relu3
);

