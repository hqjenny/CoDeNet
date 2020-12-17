#pragma once
#include <cassert>

template<typename T_IN>
void PackReadBuffer(T_IN *mem, T_IN *scale1,  T_IN *bias1, T_IN *scale3, T_IN *bias3, int OC, bool skip3, bool skip1) {
	int count = 0;
	int rep3 = skip3 ? 0 : OC;
	int rep1 = skip1 ? 0 : OC;

	for (int i = 0; i < rep1; i++) {
#pragma HLS PIPELINE
		scale1[i] = mem[count++];
	}
	for (int i = 0; i < rep1; i++) {
#pragma HLS PIPELINE
		bias1[i] = mem[count++];
	}
	for (int i = 0; i < rep3; i++) {
#pragma HLS PIPELINE
		scale3[i] = mem[count++];

	}
	for (int i = 0; i < rep3; i++) {
#pragma HLS PIPELINE
		bias3[i] = mem[count++];

	}

}

template<int PE, int W, typename T_IN, typename T_OUT>
void M2S_repeat(T_IN *mem, hls::stream<T_OUT > &s_mem, int batchD, int OCIC, bool skip) {
	int count = 0;
	int REP = (skip) ? 0 : batchD;
	for (int rep = 0; rep < REP; rep++) {
		for (int i = 0; i < OCIC; i++) {
#pragma HLS pipeline
			T_OUT w_temp;
			for (int pe = 0; pe < PE; pe++) {
				w_temp.range((pe + 1) * W - 1, pe * W) = mem[i * PE + pe];
			}
			s_mem.write(w_temp);
		}
	}
}

template<typename T_IN, typename T_OUT>
void M2S_simple(T_IN *mem, hls::stream<T_OUT > &s_mem, int REP) {
	for (int rep = 0; rep < REP; rep++) {
#pragma HLS pipeline
		s_mem.write((T_OUT)mem[rep]);
	}
}

template<typename T_IN, typename T_OUT>
void M2S_addr(T_IN *mem, hls::stream<T_OUT> &s_mem, int batch, int D, bool deform, bool skip) {
	if (deform == 0) {
		batch = 0;
	}
	for (int b = 0; b < batch; b++) {
		for (int rep = 0; rep < D * D; rep++) {
#pragma HLS pipeline
			if (skip == 0) {
				s_mem.write((T_OUT)mem[rep]);
			}
		}
	}
}





template<int PE, int in_w, int out_w, typename T_IN, typename T_OUT>
void M2S(T_IN *mem, hls::stream<T_OUT > &s_mem, int REP) {
	for (int rep = 0; rep < REP; rep++) {
#pragma HLS pipeline
		T_IN m_read = mem[rep];
		T_OUT pack;
		for (int pe = 0; pe < PE; pe++ ) {
			pack.range((pe + 1) * out_w - 1, pe * out_w) = m_read.range((pe + 1) * in_w - 1, pe * in_w);
		}
		s_mem.write(pack);
	}
}


template<int PE, int in_w, int out_w, typename T_IN, typename T_OUT>
void S2M(hls::stream<T_IN> &s_mem, T_OUT *mem, int REP) {
	int count = 0;
	for (int rep = 0; rep < REP; rep++) {
#pragma HLS pipeline II=1
		T_IN s_read = s_mem.read();
		T_OUT pack;
		for (int pe = 0; pe < PE; pe++ ) {
			pack.range((pe + 1) * out_w - 1, pe * out_w) = s_read.range((pe + 1) * in_w - 1, pe * in_w);
		}
		mem[rep] = pack;
	}
}



template<int _W_W, int PA, int PE, typename T_IN, typename T_OUT1>
void M2S_repeat_merge_1x1_v2(T_IN *mem, hls::stream<T_OUT1 > &s1, int batchD, int OCIC, bool skip1) {
	MultiChanData<PE, PA * _W_W> w_1;
#pragma HLS ARRAY_PARTITION variable=w_1.data complete dim=0
	T_IN w_read;
	int REP = skip1 ? 0 : batchD;
	for (int rep = 0; rep < REP; rep++) {
		for (int i = 0; i < OCIC; i++) {
#pragma HLS pipeline
			for (int k = 0; k < PE; k++) {
				w_1.data[k] = mem[i * PE + k];
			}
			s1.write(w_1);
		}
	}
}



template<int _W_W, int PA, int PE, typename T_IN, typename T_OUT3>
void M2S_repeat_merge_3x3(T_IN *mem, hls::stream<T_OUT3> &s3, int batchD, int OCIC, bool skip3) {
	T_IN w_read;
	MultiChanData<3 * 3, W_W * PA * PE> w_write;
	T_IN w_1;
#pragma HLS ARRAY_PARTITION variable=w_write.data complete dim=1
	for (int rep = 0; rep < batchD; rep++) {
		for (int i = 0; i < OCIC; i++) {
#pragma HLS PIPELINE
			if (skip3 == 0) {
				for (int c = 0; c < 9; c++) {
					w_read = mem[i * 9 + c];
					w_write.data[c] = w_read;
				}
				s3.write(w_write);
			}
		}
	}
}
