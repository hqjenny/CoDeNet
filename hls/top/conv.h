#ifndef __SYNTHESIS__
#include <cassert>
#endif
#include "hls_stream.h"
#include "ap_int.h"
#include<iostream>
template <unsigned int NumChannels, unsigned int DataWidth>
struct MultiChanData {
	ap_int<DataWidth> data[NumChannels];
};



template<int  _MAX_D, int _MAX_IC, int _MAX_OC, int PE, int _IN_W, int Buffer_W, typename T_IN, typename T_OUT>
void conv_3x3_M2S_new_dilav2(
		hls::stream<T_IN > &fmap,
		hls::stream<ap_uint<8> >& dila,
		hls::stream<T_OUT> &s_mem,
		ap_uint<16> D,
		ap_uint<16> IC,
		int batch,
		bool STRIDE,
		bool skip3,
		bool deform
		) {
	int index = 0;
	ap_int<16> offset_x[3];
#pragma HLS ARRAY_PARTITION variable=offset_x complete dim=0
	ap_int<16> offset_y[3];
#pragma HLS ARRAY_PARTITION variable=offset_y complete dim=0

	const int LINE_BUFFER_ROW = 15;
	T_IN line_buff[15][Buffer_W];
	int stride=(STRIDE==1)?2:1;

#pragma HLS ARRAY_PARTITION variable=line_buff complete dim=1
	int count=0;

	MultiChanData<3*3,_IN_W*PE> win;
#pragma HLS ARRAY_PARTITION variable=win.data complete dim=0
	for(int rep = 0; rep < batch; rep++){
		int ICAP = IC/PE;
		int r = -8;
		int pointer=0;
		for (ap_uint<16> i = 0; i < D+8; i++,r++) {
			int c = 0;
			for (ap_uint<16> j = 0; j < D; j++, c++) {
				bool out_start = (r >= 0) && (skip3 == 0) && (c >= 0);
				bool b_out = out_start && (r % stride==0) && (c % stride==0);
				bool read_flag = (i < D) && (j < D);

				if(read_flag){
					for(ap_uint<16> cpa=0;cpa<IC/PE;cpa++){
#pragma HLS PIPELINE
						T_IN m_read = fmap.read();
						if(skip3==1){
							win.data[0]=m_read;
							s_mem.write(win);
							continue;
						}
						else line_buff[pointer][j*ICAP+cpa] =m_read;
					}
				}
				if(b_out){
					ap_uint<8> dila_read;
					if(deform==1){
						dila_read = dila.read();
					}
					else{
						dila_read = 1;
					}
					for(ap_int<4> fi = 0; fi<3;fi++){
#pragma HLS UNROLL
						offset_y[fi] = (fi - 1) * dila_read + r;
						offset_x[fi] = (fi - 1) * dila_read + c;
						//cout<<"y"<<offset_y[fi]<<"x"<<offset_x[fi]<<endl;
					}

					for(ap_uint<16> cpa=0;cpa<IC/PE;cpa++){
#pragma HLS PIPELINE
						for(ap_uint<4> cj=0;cj<3;cj++){
							for(ap_uint<4> ci=0;ci<3;ci++){
								//cout<<offset_y[ci] <<" "<<offset_x[cj]<<"\t";
								if (offset_x[cj] < 0 || offset_y[ci] < 0 || offset_x[cj] >= D || offset_y[ci] >= D){
									win.data[ci*3+cj] = 0;
								}
								else{

									win.data[ci*3+cj]=line_buff[(offset_y[ci])%(LINE_BUFFER_ROW)][(offset_x[cj])*ICAP+cpa];
								}
							}
						}
						//cout<<endl;
						s_mem.write(win);

					}
				}
			}
			pointer++;
			if(pointer>LINE_BUFFER_ROW-1) pointer=0;
		}
	}
}

template<int  _MAX_D, int _MAX_IC, int _MAX_OC, int PE, int _IN_W, int Buffer_W, typename T_IN, typename T_OUT>
void conv_3x3_M2S(
		hls::stream<T_IN > &fmap,
		hls::stream<T_OUT> &s_mem,
		ap_uint<16> D,
		ap_uint<16> IC,
		bool STRIDE,
		bool skip3,
		bool deform
		) {
	int index = 0;
	ap_int<16> offset_x[3];
#pragma HLS ARRAY_PARTITION variable=offset_x complete dim=0
	ap_int<16> offset_y[3];
#pragma HLS ARRAY_PARTITION variable=offset_y complete dim=0

	const int LINE_BUFFER_ROW = 3;
	T_IN line_buff[3][Buffer_W];
	int stride=(STRIDE==1)?2:1;

#pragma HLS ARRAY_PARTITION variable=line_buff complete dim=1
	int count=0;

	MultiChanData<3*3,_IN_W*PE> win;
#pragma HLS ARRAY_PARTITION variable=win.data complete dim=0
	int ICAP = IC/PE;
	int r = -LINE_BUFFER_ROW+2;
	int pointer=0;
	for (ap_uint<16> i = 0; i < D+LINE_BUFFER_ROW-2; i++,r++) {
		int c = -1;
		for (ap_uint<16> j = 0; j < D+1; j++, c++) {
			bool out_start = (r >= 0) && (skip3 == 0) && (c >= 0);
			bool b_out = out_start && (r % stride==0) && (c % stride==0);
			bool read_flag = (i < D) && (j < D);

			if(read_flag){
				for(ap_uint<16> cpa=0;cpa<IC/PE;cpa++){
#pragma HLS PIPELINE
					T_IN m_read = fmap.read();
					if(skip3==1){
						win.data[0]=m_read;
						s_mem.write(win);
						continue;
					}
					else line_buff[pointer][j*ICAP+cpa] =m_read;
				}
			}
			if(b_out){
				for(ap_int<4> fi = 0; fi<3;fi++){
#pragma HLS UNROLL
					offset_y[fi] = (fi - 1) + r;
					offset_x[fi] = (fi - 1) + c;
				}

				for(ap_uint<16> cpa=0;cpa<IC/PE;cpa++){
#pragma HLS PIPELINE
					for(ap_uint<4> cj=0;cj<3;cj++){
						for(ap_uint<4> ci=0;ci<3;ci++){
							if (offset_x[cj] < 0 || offset_y[ci] < 0 || offset_x[cj] >= D || offset_y[ci] >= D){
								win.data[ci*3+cj] = 0;
							}
							else{
								win.data[ci*3+cj]=line_buff[(offset_y[ci])%(LINE_BUFFER_ROW)][(offset_x[cj])*ICAP+cpa];
							}
						}
					}
					s_mem.write(win);
				}
			}
		}
		pointer++;
		if(pointer>LINE_BUFFER_ROW-1) pointer=0;
	}

}


template<int _MAX_D,int _MAX_IC, int _MAX_OC, int PA, int PE ,int _SUM_W,int _IN_W, int _W_W>
void conv1x1_v4(
		hls::stream<ap_int<_IN_W*PA> > &fmap_in,
		hls::stream<MultiChanData<PE, _SUM_W> > &fmap_out,
		hls::stream<MultiChanData<PE, PA *_W_W> > &k0,
		int REP,
		ap_uint<16> IC,
		ap_uint<16> OC,
		bool skip1
){

	int index=0;
	ap_int<PE * _IN_W> in_value[_MAX_IC/PE];

	MultiChanData<PE, _SUM_W>  out_temp;
#pragma HLS ARRAY_PARTITION variable=out_temp.data complete dim=0
	MultiChanData<PE, _SUM_W> sum;
#pragma HLS ARRAY_PARTITION variable=sum.data complete dim=0

	for(ap_uint<8> p=0; p< PE; p++){
#pragma HLS UNROLl
		sum.data[p]=0;
	}

	for(int rep=0; rep < REP; rep++){
CPA:
		for(ap_uint<16> cpa=0; cpa < IC / PA; cpa++){
#pragma HLS PIPELINE
			ap_int<_IN_W * PA> f_read = fmap_in.read();
			if(skip1){
				for(ap_uint<8> p = 0; p < PA; p++){
#pragma HLS UNROLL
					out_temp.data[p] = (ap_int<_SUM_W>)f_read.range((p + 1) * _IN_W - 1, p * _IN_W);
				}
				fmap_out.write(out_temp);
				continue;
			}
			else{
				in_value[cpa] = f_read;
			}
		}

OUT_CHANNEL:
		for(ap_uint<16> k=0;k<OC/PE;k++){
			if(skip1) break;
			for(ap_uint<16> cpa=0; cpa < IC/PA; cpa++){
#pragma HLS PIPELINE
				ap_int<_IN_W * PA> win = in_value[cpa];
				MultiChanData< PE, PA *_W_W> w_vec;
#pragma HLS ARRAY_PARTITION variable=w_vec.data complete dim=0
				w_vec = k0.read();

PAPE:			for(ap_uint<8> pe = 0; pe < PE; pe++){
#pragma HLS UNROLL
					for(ap_uint<8> pa = 0; pa < PA; pa++){
#pragma HLS UNROLL
						ap_int<_SUM_W> partial_sum = 0;
						partial_sum += (ap_int<_IN_W>) win.range(_IN_W*(pa+1)-1,_IN_W*pa) * (ap_int<_W_W>) w_vec.data[pe].range((pa + 1) * _W_W - 1, pa * _W_W);
						ap_int<_SUM_W> old_sum = sum.data[pe];
						partial_sum += old_sum;
						sum.data[pe] = partial_sum;
						//cout<<" in "<<(ap_int<_IN_W>) win.range(_IN_W*(pa+1)-1,_IN_W*pa) <<" w "<<(ap_int<_W_W>) w_vec.data[pe].range((pa + 1) * _W_W - 1, pa * _W_W);
					}
					//cout<<endl;
				}

			}
			fmap_out<<sum;
			for(ap_uint<8> p=0; p<PE; p++){
#pragma HLS UNROLL
				sum.data[p]=0;
			}
		}
	}

}


template<int _IN_W, int ACT_W, int PA, int SCALE_W, int BIAS_W, int _SHIFT_W>
void quantize_mul_shift(
	hls::stream<MultiChanData<PA, _IN_W> > &fmap,
	hls::stream<ap_int<PA * ACT_W> > &out,
	hls::stream<ap_int<PA * SCALE_W> > &s_scale,
	hls::stream<ap_int<PA * BIAS_W> > &s_bias,
	int rep,
	ap_uint<16> C,
	bool skip,
	bool relu
	){
	//static int count=0;
	int CPA = C/PA;
	int upper = (1 << (ACT_W - 1)) - 1;
	int lower = -(1 << (ACT_W - 1));
	int neg_shift = 1 << (_SHIFT_W - 1);

	for(int bat = 0; bat < rep; bat++){
		for(ap_uint<16> k=0;k<CPA;k++){
#pragma HLS PIPELINE II=1
			MultiChanData<PA, _IN_W> input = fmap.read();
#pragma HLS ARRAY_PARTITION variable=input.data complete dim=0

			ap_int<PA * ACT_W> value;
			if(skip==1){
				for(ap_uint<8> pi=0;pi<PA;pi++){
					value.range((pi + 1) * ACT_W - 1, pi * ACT_W) = (ap_int<ACT_W>)input.data[pi];
				}
				out.write(value);
				continue;
			}
			ap_int<PA * SCALE_W> scale = s_scale.read();
			ap_int<PA * BIAS_W> bias = s_bias.read();
			for(ap_uint<8> pa=0;pa<PA;pa++){
				ap_int<_IN_W> part = input.data[pa];
				ap_int<SCALE_W> s = scale.range((pa + 1) * SCALE_W - 1, pa * SCALE_W);
				ap_int<BIAS_W> b =  bias.range((pa + 1) * BIAS_W - 1, pa * BIAS_W);
				ap_int<SCALE_W + _IN_W> partial_result = part * s;
				if(partial_result < 0){
					partial_result += neg_shift;
				}
				ap_int<SCALE_W + _IN_W - _SHIFT_W> quant = partial_result.range(SCALE_W + _IN_W - 1, _SHIFT_W);
				quant += b;
				//cout<<count++<<"partial_sum "<<(int)part<<"scale "<<s<<"bias "<<b<<"partial "<<partial_result<<"quant "<<quant;
				if(relu){
					if(quant<0) quant=0;
					quant -= 1 << (ACT_W - 1);
				}
				if(quant > upper) quant = upper;
				else if(quant < lower) quant = lower;
				value.range((pa + 1) * ACT_W - 1, pa * ACT_W) = (ap_int<ACT_W>) quant;
				//cout<<"out"<<(ap_int<ACT_W>) quant<<endl;
			}
			out.write(value);
		}
	}
}



template<int MAX_d,int _MAX_C,  int PE ,int _SUM_W,int _IN_W, int _W_W,int FIRST_C>
void dw_deform_M(
		hls::stream<MultiChanData<3*3,_IN_W*PE> > &fmap_in,
		hls::stream<MultiChanData<PE, _SUM_W> > &fmap_out,
		hls::stream<MultiChanData<3*3,_W_W*PE> > &k0,
		ap_uint<16> D,
		ap_uint<16> IC,
		int batch,
		bool STRIDE_2,
		bool skip3
){


	int count=0;
	const ap_uint<4> K_SIZE = 3;
	ap_uint<4> STRIDE = (STRIDE_2==1)? 2:1;
	int ICAP = IC/PE;

	for(int rep = 0; rep < batch; rep++){
WRITE_ROW_LOOP:
		for(ap_uint<16> row=0;row<D;row++){
WRITE_COW_LOOP:
			for(ap_uint<16> col=0;col<D;col++){
CPA:
				bool b_out = (row % STRIDE == 0) && (col % STRIDE == 0);
				if(b_out){
					for(ap_uint<16> cpa=0;cpa<_MAX_C/PE;cpa++){
#pragma HLS PIPELINE
						if(cpa>=ICAP) break;
						MultiChanData<9,_W_W*PE> w_vec;
#pragma HLS ARRAY_PARTITION variable=w_vec.data complete dim=0
						if(skip3==0){
							w_vec= k0.read();
						}
						MultiChanData<9,_IN_W*PE> win= fmap_in.read();
#pragma HLS ARRAY_PARTITION variable=win.data complete dim=0

						MultiChanData<PE, _SUM_W> out_sum;
#pragma HLS ARRAY_PARTITION variable=out_sum.data complete dim=0

PAPE:				if(skip3==0){
							for(ap_uint<8> pe=0;pe<PE;pe++){
#pragma HLS UNROLL
								ap_int<_SUM_W> partial_sum = 0;
								int w_start = 0;
								int w_end = _W_W-1;
IN_ROW:
								for(ap_uint<4> r=0;r<K_SIZE;r++){
#pragma HLS UNROLL
IN_COL:
									for(ap_uint<4> c=0;c<K_SIZE;c++){
#pragma HLS UNROLL
										partial_sum += (ap_int<_IN_W>) win.data[r*K_SIZE+c].range(_IN_W*(pe+1)-1,_IN_W*pe) * (ap_int<_W_W>) w_vec.data[r*K_SIZE+c].range((pe+1)*_W_W-1,pe*_W_W);
										//cout<<" in "<<(ap_int<_IN_W>) win.data[r*K_SIZE+c].range(_IN_W*(pe+1)-1,_IN_W*pe);
										//cout<<" w "<<(ap_int<_W_W>) w_vec.data[r*K_SIZE+c].range((pe+1)*_W_W-1,pe*_W_W);
									}
								}
								//cout<<" p "<<partial_sum<<endl;
								out_sum.data[pe] = partial_sum;
							}
							fmap_out<<out_sum;
						}
						else{
							for(ap_uint<8> p=0; p<PE; p++){
								out_sum.data[p] = (ap_int<_SUM_W>)win.data[0].range((p + 1) * _IN_W - 1, p * _IN_W);
							}
							ap_int<PE*_SUM_W> out_temp = (ap_int<PE*_SUM_W>) win.data[0];
							fmap_out<<out_sum;
						}
					}
				}
			}
		}
	}

}


