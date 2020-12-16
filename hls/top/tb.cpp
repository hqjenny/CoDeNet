#include "deform.h"

template<int W, int PA, typename T_OUT>
void readfilepack(T_OUT *mem, int length, const char* name){
	static int count = 0;
	FILE *f_s;
	f_s=fopen(name,"r");
	for (int i=0; i<length/PA; i++){
	  T_OUT temp;
	  for(int j=0;j<PA;j++){
		  int tmp;
		  fscanf(f_s, "%d", &tmp);
		  temp.range(W*(j+1)-1,W*j)=tmp;
	  }
	  mem[count++] = temp;
	  //cout<<endl;
	}
	fclose(f_s);
}
template<int W, int PA, typename T_OUT>
void readfile(T_OUT *mem, int length, const char* name){
	int count = 0;
	FILE *f_s;
	f_s=fopen(name,"r");
	for (int i=0; i<length/PA; i++){
	  T_OUT temp;
	  for(int j=0;j<PA;j++){
		  int tmp;
		  fscanf(f_s, "%d", &tmp);
		  temp.range(W*(j+1)-1,W*j)=tmp;

	  }
	  mem[count++] = temp;
	  //cout<<endl;
	}
	fclose(f_s);
}



int main(){
	int retval=0;
	const int batch =1;
	//const int FM_D=224;
	const int D=16;
	const int IC=512;
	const int OC = 512;
	bool relu1=1;
	bool relu3=1;

	static ap_int<IN_W*PE_3> fmap_in[batch*D*D*IC/PE_3];
	static ap_int<S_W> quant[OC*4];


	static ap_int<W_W * PA_0> wtemp1[IC*IC];
	static ap_int<W_W*1*PE_3> wtemp3[10*9*(IC/1)/PE_3];

	static ap_uint<8> addr[D*D];
	for(int i=0;i<D*D;i++){
		addr[i] = 7;
	}
	const int OUT_NUM = (batch*D*D*OC);
	static ap_int<OUT_W*PE_0> out[8*OUT_NUM/PE_0];


	readfile<FM_W, PA_0>(fmap_in, D*D*IC, "D16_IC512_OC512_ksize3_dila7input.txt");
	readfilepack<S_W, 1>(quant, OC*2, "D16_IC512_OC512_ksize3_dila7quant.txt");
	readfile<W_W, PA_0>(wtemp3,  IC * 9, "D16_IC512_OC512_ksize3_dila7weight3.txt");

//	readfile<FM_W, PA_0>(fmap_in, D*D*IC, "D4_IC512_OC512_ksize1input.txt");
//	readfilepack<S_W, 1>(quant, OC*2, "D4_IC512_OC512_ksize1quant.txt");
//	readfile<W_W, PA_0>(wtemp1,  IC * OC, "D4_IC512_OC512_ksize1weight1.txt");

	bool skip1=1;
	bool skip3=0;
	bool stride_2 = 0;
	bool deform = 1;

    top(fmap_in, out, wtemp1, wtemp3, quant,addr, D, IC, OC, batch, stride_2,skip3,skip1,deform,relu1,relu3);


    FILE *f_gt;

	f_gt = fopen("D16_IC512_OC512_ksize3_dila7output.txt","r");

	int error_count=0;
	int count=0;
    for (int i=0; i<OUT_NUM/PE_0; i++){
    	ap_uint<FM_W*PE_0> rd = out[i];
    	for(int j=0;j<PE_0;j++){
    		int out_read = (ap_int<FM_W>) (rd.range((j+1)*4-1,j*4));
    		int gt;
    		fscanf(f_gt, "%d", &gt);
    		if (gt!=out_read){
    			cout<<count<<"\t"<<error_count++<<"\t"<<gt<<"\t"<<out_read<<endl;
    		}
    		count++;

    	}
    }
    fclose(f_gt);






    return 0;
}




