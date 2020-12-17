#include "deform.h"

template<int W, int PA, typename T_OUT>
void readfilepack(T_OUT *mem, int length, const char* name) {
	static int count = 0;
	FILE *f_s;
	f_s = fopen(name, "r");
	for (int i = 0; i < length / PA; i++) {
		T_OUT temp;
		for (int j = 0; j < PA; j++) {
			int tmp;
			fscanf(f_s, "%d", &tmp);
			temp.range(W * (j + 1) - 1, W * j) = tmp;
		}
		mem[count++] = temp;
	}
	fclose(f_s);
}

template<int W, int PA, typename T_OUT>
void readfile(T_OUT *mem, int length, const char* name) {
	int count = 0;
	FILE *f_s;
	f_s = fopen(name, "r");
	for (int i = 0; i < length / PA; i++) {
		T_OUT temp;
		for (int j = 0; j < PA; j++) {
			int tmp;
			fscanf(f_s, "%d", &tmp);
			temp.range(W * (j + 1) - 1, W * j) = tmp;

		}
		mem[count++] = temp;
	}
	fclose(f_s);
}



int main() {
	int retval = 0;
	const int batch = 1;
	const int D = 32;
	const int IC = 64;
	const int OC = 16;
	bool relu1 = 1;
	bool relu3 = 1;

	static ap_int<IN_W*PE_3> fmap_in[batch * D * D * IC / PE_3];
	static ap_int<S_W> quant[OC * 4];
	static ap_int<W_W * PA_0> wtemp1[IC * OC];
	static ap_int<W_W * 1 * PE_3> wtemp3[10 * 9 * (IC / 1) / PE_3];

	static ap_uint<8> offset[D * D];
	for (int i = 0; i < D * D; i++) {
		offset[i] = 1;
	}
	const int OUT_NUM = (batch * D * D * OC);
	static ap_int<OUT_W*PE_0> out[8 * OUT_NUM / PE_0];


//	readfile<FM_W, PA_0>(fmap_in, D * D * IC, "D_32_C_64_k_3_PA_16_PE_16input.txt");
//	readfilepack<S_W, 1>(quant, OC * 2, "D_32_C_64_k_3_PA_16_PE_16quant.txt");
//	readfile<W_W, PA_0>(wtemp3,  IC * 9, "D_32_C_64_k_3_PA_16_PE_16weight3.txt");

	readfile<FM_W, PA_0>(fmap_in, D*D*IC, "D_32_IC_64_OC_16_k_1_PA_16_PE_16input.txt");
	readfilepack<S_W, 1>(quant, OC*2, "D_32_IC_64_OC_16_k_1_PA_16_PE_16quant.txt");
	readfile<W_W, PA_0>(wtemp1,  IC * OC, "D_32_IC_64_OC_16_k_1_PA_16_PE_16weight1.txt");

	bool skip1 = 0;
	bool skip3 = 1;
	bool stride_2 = 0;
	bool deform = 0;

	top(fmap_in, out, wtemp1, wtemp3, quant, offset, D, IC, OC, batch, stride_2, skip3, skip1, deform, relu1, relu3);


	FILE *f_gt;

	//f_gt = fopen("D_32_C_64_k_3_PA_16_PE_16output.txt", "r");
	f_gt = fopen("D_32_IC_64_OC_16_k_1_PA_16_PE_16output.txt", "r");

	int error_count = 0;
	int count = 0;
	for (int i = 0; i < OUT_NUM / PE_0; i++) {
		ap_uint<FM_W*PE_0> rd = out[i];
		for (int j = 0; j < PE_0; j++) {
			ap_int<FM_W> out_read = (ap_int<FM_W>) rd.range((j + 1) * FM_W - 1, FM_W * j);
			int gt;
			fscanf(f_gt, "%d", &gt);
			if (gt != out_read) {
				cout << count << "\t" << error_count++ << "\t" << gt << "\t" << out_read << endl;
			}
			count++;

		}
	}
	fclose(f_gt);

	return 0;
}




