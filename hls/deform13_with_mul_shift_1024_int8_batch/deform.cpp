#include "deform.h"

using namespace std;


template<int PA, int PE, int _MAX_IC, int _MAX_OC, int _MAX_D, int _IN_W, int _SUM_W,int _W_W, int _S_W, int _B_W>
void conv1x1_pack(
		hls::stream<ap_int<_IN_W*PA> > &fmap_in,
		hls::stream<ap_int<_IN_W*PE> > &fmap_out,
		ap_int<_W_W * PA> *k0,
		ap_int<_S_W> *scale,
		ap_int<_B_W> *bias,
		ap_uint<16> D,
		ap_uint<16> IC,
		ap_uint<16> OC,
		int batch,
		bool skip1,
		bool relu
){
#pragma HLS INLINE
	 hls::stream<MultiChanData<PE, _SUM_W> > s_conv1;
#pragma HLS STREAM variable=s_conv1 depth=2
	 hls::stream<MultiChanData<PE, PA * W_W> > k1;
#pragma HLS STREAM variable=k1 depth=2
	hls::stream<ap_int<PE*_S_W> > s_scale;
#pragma HLS STREAM variable=s_scale depth=2
	hls::stream<ap_int<PE*_B_W> > s_bias;
#pragma HLS STREAM variable=s_bias depth=2


	M2S_repeat<PE, _S_W>(scale, s_scale, batch*D*D, OC/PE,skip1);
	M2S_repeat<PE, _B_W>(bias, s_bias, batch*D*D, OC/PE,skip1);

	ap_int<_W_W * PA> k1_buffer[_MAX_IC * _MAX_OC/PA];
#pragma HLS ARRAY_PARTITION variable=k1_buffer cyclic factor=16


	for (int i=0; i < OC * IC / PA; i++) {
#pragma HLS PIPELINE
		k1_buffer[i] = k0[i];
	}

	M2S_repeat_merge_1x1_v2<_W_W, PA, PE>(k1_buffer, k1, batch*D*D, OC*IC/PA/PE, skip1);
	conv1x1_v4<_MAX_D,_MAX_IC,_MAX_OC,PA,PE,_SUM_W,_IN_W,_W_W>(fmap_in,s_conv1,k1,D*D*batch,IC,OC,skip1);
	quantize_mul_shift<_SUM_W, _IN_W, PE, _S_W, _B_W, SHIFT_W0 >(s_conv1, fmap_out, s_scale, s_bias, D*D*batch, OC, skip1, relu);
}

template<int PE, int _MAX_C, int _MAX_D, int _IN_W, int _SUM_W,int _W_W, int _S_W, int _B_W>
void conv3x3_pack(
		hls::stream<ap_int<_IN_W*PE> > &fmap_in,
		hls::stream<ap_int<PE*_IN_W> > &fmap_out,
		hls::stream<ap_uint<8> > &addr,
		ap_int<_W_W * PE> *k3,
		ap_int<_S_W> *scale,
		ap_int<_B_W> *bias,
		ap_uint<16> D,
		ap_uint<16> IC,
		int batch,
		bool STRIDE,
		bool skip3,
		bool deform,
		bool relu
){
#pragma HLS INLINE
	int CONV_D =STRIDE ? D >> 1 : D;
	 hls::stream<MultiChanData<3*3,_IN_W*PE> > in_layer;
#pragma HLS STREAM variable=in_layer depth=2
	 hls::stream<MultiChanData<9,_W_W*PE> > k3s;
#pragma HLS STREAM variable=k3s depth=2
	 hls::stream<MultiChanData<PE, _SUM_W> > s_conv3;
#pragma HLS STREAM variable=s_conv3 depth=2
	hls::stream<ap_int<PE *_S_W> > s_scale;
#pragma HLS STREAM variable=s_scale depth=2
	hls::stream<ap_int<PE * _B_W> > s_bias;
#pragma HLS STREAM variable=s_bias depth=2


	ap_int<W_W * PE_3> k3_buffer[9 * _MAX_C / (PE)];
#pragma HLS ARRAY_PARTITION variable=k3_buffer cyclic factor=9

	for (int i=0; i <9* IC /(PE); i++) {
		k3_buffer[i] = k3[i];
	}


	M2S_repeat<PE, _S_W>(scale, s_scale, batch*CONV_D*CONV_D, IC/PE,skip3);
	M2S_repeat<PE, _B_W>(bias, s_bias, batch*CONV_D*CONV_D, IC/PE,skip3);
	M2S_repeat_merge_3x3<_W_W, 1, PE, ap_int<_W_W*PE>,MultiChanData<9,_W_W*PE> >(k3_buffer, k3s, batch*CONV_D*CONV_D, IC/PE,skip3);
	conv_3x3_M2S_new_dilav2<_MAX_D, _MAX_C, _MAX_C, PE,_IN_W, _MAX_C * 32 / PE>(fmap_in,addr, in_layer, D, IC, batch, STRIDE, skip3,deform);
	dw_deform_M<_MAX_D,_MAX_C,PE,_SUM_W,_IN_W,_W_W,1>(in_layer,s_conv3,k3s,D,IC,batch,STRIDE, skip3);
	quantize_mul_shift<_SUM_W, _IN_W, PE, _S_W, _B_W, SHIFT_W0>(s_conv3, fmap_out, s_scale, s_bias, batch * CONV_D * CONV_D, IC, skip3, relu );
}

void wrapper(
	ap_int<IN_W * PE_3>* fmap,
	ap_int<OUT_W * PE_0> * out,
	ap_int<W_W * PA_0>*k1,
	ap_int<W_W * PE_3>*k3,
	ap_int<S_W>*quant,
	ap_uint<8>* dilation,
	int FM_D,
	int IC,
	int OC,
	int batch,
	bool STRIDE_2,
	bool skip3,
	bool skip1,
	bool deform,
	bool relu1,
	bool relu3
	){
#pragma HLS INLINE




	int CONV_D =STRIDE_2 ? FM_D >> 1 : FM_D;

	 hls::stream<ap_int<FM_W*PE_3> > fin;
#pragma HLS STREAM variable=fin depth=2 dim=1
	 hls::stream<ap_int<FM_W*PE_3> > f1;
#pragma HLS STREAM variable=f1 depth=2 dim=1
	 hls::stream<ap_int<FM_W*PE_3> > f3;
#pragma HLS STREAM variable=f3 depth=2 dim=1
	 hls::stream<ap_uint<8> > dila;
#pragma HLS STREAM variable=dila depth=2 dim=1


	ap_int<S_W> scale_buffer3[ MAX_C];
	ap_int<B_W> bias_buffer3[ MAX_C];
	ap_int<S_W> scale_buffer1[ MAX_C];
	ap_int<B_W> bias_buffer1[ MAX_C];
#pragma HLS ARRAY_PARTITION variable=scale_buffer3 cyclic factor=8
#pragma HLS ARRAY_PARTITION variable=bias_buffer3 cyclic factor=8
#pragma HLS ARRAY_PARTITION variable=scale_buffer1 cyclic factor=8
#pragma HLS ARRAY_PARTITION variable=bias_buffer1 cyclic factor=8



	M2S_addr(dilation, dila, batch, CONV_D ,deform,skip3);

	M2S<PA_0, IN_W, FM_W>(fmap, fin, batch * FM_D * FM_D * IC / PE_3);
	PackReadBuffer(quant, scale_buffer1, bias_buffer1, scale_buffer3,  bias_buffer3, OC, skip3, skip1);

	conv1x1_pack<PE_3,PE_0,MAX_IC,MAX_OC,MAX_D,FM_W,SUM_W,W_W,S_W,B_W>(fin,f1,k1,scale_buffer1,bias_buffer1,FM_D,IC,OC,batch, skip1,relu1);
	conv3x3_pack<PE_3,MAX_OC,MAX_D,FM_W,SUM_W,W_W,S_W,B_W>(f1, f3, dila, k3, scale_buffer3, bias_buffer3, FM_D, OC, batch, STRIDE_2, skip3, deform, relu3);

	S2M<PE_0, FM_W, OUT_W>(f3, out, batch * CONV_D*CONV_D*OC/PE_0);


}

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
){


#pragma HLS INTERFACE m_axi port=fmap offset=slave bundle=gmem0 depth=2
#pragma HLS INTERFACE m_axi port=out offset=slave bundle=gmem1 depth=2
#pragma HLS INTERFACE m_axi port=k0_1 bundle=gmem2 depth=2
#pragma HLS INTERFACE m_axi port=k0_3 bundle=gmem3 depth=2
#pragma HLS INTERFACE m_axi port=quant bundle=gmem4 depth=2
#pragma HLS INTERFACE m_axi port=dilation bundle=gmem5 depth=2
//#endif

#pragma HLS INTERFACE s_axilite port=fmap bundle=control 
#pragma HLS INTERFACE s_axilite port=out bundle=control
#pragma HLS INTERFACE s_axilite port=k0_1 bundle=control
#pragma HLS INTERFACE s_axilite port=k0_3 bundle=control
#pragma HLS INTERFACE s_axilite port=quant bundle=control
#pragma HLS INTERFACE s_axilite port=D bundle=control
#pragma HLS INTERFACE s_axilite port=IC bundle=control
#pragma HLS INTERFACE s_axilite port=OC bundle=control
#pragma HLS INTERFACE s_axilite port=STRIDE_2 bundle=control
#pragma HLS INTERFACE s_axilite port=skip3 bundle=control
#pragma HLS INTERFACE s_axilite port=skip1 bundle=control
#pragma HLS INTERFACE s_axilite port=dilation bundle=control
#pragma HLS INTERFACE s_axilite port=deform bundle=control
#pragma HLS INTERFACE s_axilite port=relu1 bundle=control
#pragma HLS INTERFACE s_axilite port=relu3 bundle=control
#pragma HLS INTERFACE s_axilite port=batch bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

#pragma HLS DATAFLOW
	wrapper(fmap, out, k0_1,k0_3,quant, dilation,D, IC, OC, batch, STRIDE_2, skip3, skip1, deform, relu1, relu3);

}
