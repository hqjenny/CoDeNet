#ifndef __SYNTHESIS__
#include <cassert>
#endif
#include "hls_stream.h"
#include "ap_int.h"
#include<iostream>

// Template for a multi-channel data array with specific
// bit-width and channel size. It can be used as the data
// structure for dataflow fifo.
template <unsigned int NumChannels, unsigned int DataWidth>
struct MultiChanData {
	ap_int<DataWidth> data[NumChannels];
};


// Template for conv 3x3 linebuffer with 15 line for deformable
// input buffer. It will buffer feature map and read a deformable
// sampling windows (square shape) using the offset values. This offset will
// be fixed to 1 for ordinary conv3x3.
template<int _MAX_D, int _MAX_IC, int _MAX_OC, int PE, int _IN_W, int Buffer_W, typename T_IN, typename T_OUT>
void conv_3x3_linebuffer(
    hls::stream<T_IN > &fmap,
    hls::stream<ap_uint<8> >& offset,
    hls::stream<T_OUT> &s_mem,
    ap_uint<16> D,
    ap_uint<16> IC,
    int batch,
    bool STRIDE,
    bool skip3,
    bool deform
) {

	ap_int<16> offset_x[3];
	ap_int<16> offset_y[3];
#pragma HLS ARRAY_PARTITION variable=offset_x complete dim=0
#pragma HLS ARRAY_PARTITION variable=offset_y complete dim=0

	const int LINE_BUFFER_ROW = 15;
	T_IN line_buff[15][Buffer_W];
	int stride = (STRIDE == 1) ? 2 : 1;
#pragma HLS ARRAY_PARTITION variable=line_buff complete dim=1


	MultiChanData<3 * 3, _IN_W*PE> win;
#pragma HLS ARRAY_PARTITION variable=win.data complete dim=0
	for (int rep = 0; rep < batch; rep++) {
		int ICAP = IC / PE;
		int r = -8;
		int pointer = 0;
		for (ap_uint<16> i = 0; i < D + 8; i++, r++) { // first buffer 8 line
			int c = 0;
			for (ap_uint<16> j = 0; j < D; j++, c++) {
				bool out_start = (r >= 0) && (skip3 == 0) && (c >= 0);
				bool out_flag = out_start && (r % stride == 0) && (c % stride == 0);
				bool read_flag = (i < D) && (j < D);

				if (read_flag) {
					for (ap_uint<16> cpa = 0; cpa < IC / PE; cpa++) {
#pragma HLS PIPELINE
						T_IN m_read = fmap.read();
						if (skip3 == 1) {  // if skip 3x3, directly output the result
							win.data[0] = m_read;
							s_mem.write(win);
							continue;
						}
						else line_buff[pointer][j * ICAP + cpa] = m_read;
					}
				}
				if (out_flag) {
					ap_uint<8> offset_read;
					if (deform == 1) {
						offset_read = offset.read();
					}
					else {
						offset_read = 1;
					}
					for (ap_int<4> fi = 0; fi < 3; fi++) {
#pragma HLS UNROLL
						// generate offsets of 9 points using the offset value
						offset_y[fi] = (fi - 1) * offset_read + r;
						offset_x[fi] = (fi - 1) * offset_read + c;
					}

					for (ap_uint<16> cpa = 0; cpa < IC / PE; cpa++) {
#pragma HLS PIPELINE
						for (ap_uint<4> cj = 0; cj < 3; cj++) {
							for (ap_uint<4> ci = 0; ci < 3; ci++) {
								if (offset_x[cj] < 0 || offset_y[ci] < 0 || offset_x[cj] >= D || offset_y[ci] >= D) {
									// padding region
									ap_int<_IN_W * PE> non_zero_pad = 0;
									for (ap_uint<8> pn = 0; pn < PE; pn ++){
										non_zero_pad.range((pn + 1) * _IN_W - 1, pn * _IN_W) = (ap_int<_IN_W>) -128;
									}
									win.data[ci * 3 + cj] = non_zero_pad;
								}
								else {
									// non padding region
									win.data[ci * 3 + cj] = line_buff[(offset_y[ci]) % (LINE_BUFFER_ROW)][(offset_x[cj]) * ICAP + cpa];
								}
							}
						}
						s_mem.write(win);

					}
				}
			}
			// pointer to the current row for buffering
			pointer++;
			if (pointer > LINE_BUFFER_ROW - 1) pointer = 0;
		}
	}
}

// Templates for conv1x1
template<int _MAX_D, int _MAX_IC, int _MAX_OC, int PA, int PE , int _SUM_W, int _IN_W, int _W_W>
void conv1x1_v4(
    hls::stream<ap_int<_IN_W*PA> > &fmap_in,
    hls::stream<MultiChanData<PE, _SUM_W> > &fmap_out,
    hls::stream<MultiChanData<PE, PA *_W_W> > &k0,
    int REP,
    ap_uint<16> IC,
    ap_uint<16> OC,
    bool skip1
) {

	ap_int<PE * _IN_W> in_value[_MAX_IC / PE];
	MultiChanData< PE, PA *_W_W> w_vec;
	MultiChanData<PE, _SUM_W>  out_temp;
	MultiChanData<PE, _SUM_W> sum;
#pragma HLS ARRAY_PARTITION variable=out_temp.data complete dim=0
#pragma HLS ARRAY_PARTITION variable=sum.data complete dim=0
#pragma HLS ARRAY_PARTITION variable=w_vec.data complete dim=0

	for (ap_uint<8> p = 0; p < PE; p++) {
#pragma HLS UNROLl
		sum.data[p] = 0;
	}

	for (int rep = 0; rep < REP; rep++) {
		for (ap_uint<16> cpa = 0; cpa < IC / PA; cpa++) { // buffering one position of input
#pragma HLS PIPELINE
			ap_int<_IN_W * PA> f_read = fmap_in.read();
			if (skip1) {
				for (ap_uint<8> p = 0; p < PA; p++) {
#pragma HLS UNROLL
					out_temp.data[p] = (ap_int<_SUM_W>)f_read.range((p + 1) * _IN_W - 1, p * _IN_W);
				}
				fmap_out.write(out_temp);
				continue;
			}
			else {
				in_value[cpa] = f_read;
			}
		}

		for (ap_uint<16> k = 0; k < OC / PE; k++) {
			if (skip1) break;
			for (ap_uint<16> cpa = 0; cpa < IC / PA; cpa++) {
#pragma HLS PIPELINE
				ap_int<_IN_W * PA> win = in_value[cpa];
				w_vec = k0.read();

				// PA * PE MAC units
				for (ap_uint<8> pe = 0; pe < PE; pe++) {
#pragma HLS UNROLL
					for (ap_uint<8> pa = 0; pa < PA; pa++) {
#pragma HLS UNROLL
						ap_int<_SUM_W> partial_sum = 0;
						partial_sum += (ap_int<_IN_W>) win.range(_IN_W * (pa + 1) - 1, _IN_W * pa) * (ap_int<_W_W>) w_vec.data[pe].range((pa + 1) * _W_W - 1, pa * _W_W);
						ap_int<_SUM_W> old_sum = sum.data[pe];
						partial_sum += old_sum;
						sum.data[pe] = partial_sum;
					}
				}
			}
			fmap_out << sum;
			for (ap_uint<8> p = 0; p < PE; p++) {
#pragma HLS UNROLL
				sum.data[p] = 0;
			}
		}
	}
}

// Depthwise 3x3 compute unit
template<int MAX_d, int _MAX_C,  int PE , int _SUM_W, int _IN_W, int _W_W, int FIRST_C>
void dw_deform_M(
    hls::stream<MultiChanData<3 * 3, _IN_W*PE> > &fmap_in,
    hls::stream<MultiChanData<PE, _SUM_W> > &fmap_out,
    hls::stream<MultiChanData<3 * 3, _W_W*PE> > &k0,
    ap_uint<16> D,
    ap_uint<16> IC,
    int batch,
    bool STRIDE_2,
    bool skip3
) {

	const ap_uint<4> K_SIZE = 3;
	ap_uint<4> STRIDE = (STRIDE_2 == 1) ? 2 : 1;
	int ICAP = IC / PE;

	MultiChanData<9, _W_W*PE> w_vec;
	MultiChanData<9, _IN_W*PE> win;
	MultiChanData<PE, _SUM_W> out_sum;
#pragma HLS ARRAY_PARTITION variable=w_vec.data complete dim=0
#pragma HLS ARRAY_PARTITION variable=win.data complete dim=0
#pragma HLS ARRAY_PARTITION variable=out_sum.data complete dim=0

	for (int rep = 0; rep < batch; rep++) {
		for (ap_uint<16> row = 0; row < D; row++) {
			for (ap_uint<16> col = 0; col < D; col++) {
				bool out_flag = (row % STRIDE == 0) && (col % STRIDE == 0);
				if (out_flag) {
					for (ap_uint<16> cpa = 0; cpa < _MAX_C / PE; cpa++) {
#pragma HLS PIPELINE
						if (cpa >= ICAP) break;

						if (skip3 == 0) {
							w_vec = k0.read();
						}
						win = fmap_in.read();
						if (skip3 == 0) {  // 3 * 3 * PE MAC units
							for (ap_uint<8> pe = 0; pe < PE; pe++) {
#pragma HLS UNROLL
								ap_int<_SUM_W> partial_sum = 0;
								for (ap_uint<4> r = 0; r < K_SIZE; r++) {
#pragma HLS UNROLL
									for (ap_uint<4> c = 0; c < K_SIZE; c++) {
#pragma HLS UNROLL
										partial_sum += (ap_int<_IN_W>) win.data[r * K_SIZE + c].range(_IN_W * (pe + 1) - 1, _IN_W * pe) * (ap_int<_W_W>) w_vec.data[r * K_SIZE + c].range((pe + 1) * _W_W - 1, pe * _W_W);
									}
								}
								out_sum.data[pe] = partial_sum;
							}
							fmap_out << out_sum;
						}
						else {  // direct output if skip 3x3
							for (ap_uint<8> p = 0; p < PE; p++) {
								out_sum.data[p] = (ap_int<_SUM_W>)win.data[0].range((p + 1) * _IN_W - 1, p * _IN_W);
							}
							ap_int<PE*_SUM_W> out_temp = (ap_int<PE*_SUM_W>) win.data[0];
							fmap_out << out_sum;
						}
					}
				}
			}
		}
	}
}


// templeta for quantization unit
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
) {

	int count = 0;
	int CPA = C / PA;
	int upper = (1 << (ACT_W - 1)) - 1;
	int lower = -(1 << (ACT_W - 1));
	const int round_shift = 1 << (_SHIFT_W - 1);

	for (int bat = 0; bat < rep; bat++) {
		for (ap_uint<16> k = 0; k < CPA; k++) {
#pragma HLS PIPELINE II=1
			MultiChanData<PA, _IN_W> input = fmap.read();
#pragma HLS ARRAY_PARTITION variable=input.data complete dim=0

			ap_int<PA * ACT_W> value;
			if (skip == 1) {
				for (ap_uint<8> pi = 0; pi < PA; pi++) {
					value.range((pi + 1) * ACT_W - 1, pi * ACT_W) = (ap_int<ACT_W>)input.data[pi];
				}
				out.write(value);
				continue;
			}
			ap_int<PA * SCALE_W> scale = s_scale.read();
			ap_int<PA * BIAS_W> bias = s_bias.read();

			for (ap_uint<8> pa = 0; pa < PA; pa++) {
				ap_int<_IN_W> part = input.data[pa];
				ap_int<SCALE_W> s = scale.range((pa + 1) * SCALE_W - 1, pa * SCALE_W);
				ap_int<BIAS_W> b =  bias.range((pa + 1) * BIAS_W - 1, pa * BIAS_W);
				ap_int < SCALE_W + _IN_W > partial_result = part * s; // multiply partial sum with scale
				partial_result += round_shift; //  result shift to perform mathematical rounding
				ap_int < SCALE_W + _IN_W - _SHIFT_W > quant = partial_result.range(SCALE_W + _IN_W - 1, _SHIFT_W);
				quant += b; // add bias
				if (relu) {
					if (quant < 0) quant = 0;
				}
				quant -= 1 << (ACT_W - 1); // asymmetric quantization zero shift

				// clamp range
				if (quant > upper) quant = upper;
				else if (quant < lower) quant = lower;

				value.range((pa + 1) * ACT_W - 1, pa * ACT_W) = (ap_int<ACT_W>) quant;
			}
			out.write(value);
		}
	}
}




