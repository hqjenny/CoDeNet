# HLS Project

The source code of the HLS-based accelerator is under [top](top) subdirection. 
We also provide an archive of the HLS project containing the CoDeNet accelerator code [here](https://people.eecs.berkeley.edu/~qijing.huang/2021FPGA/CoDeNet_hls.zip).

Please unzip the file `unzip CoDeNet_hls.zip`, open Vivado HLS GUI, and load the project. 

The Vivado Project and Block Design files can be downloaded [here](https://people.eecs.berkeley.edu/~qijing.huang/2021FPGA/CoDeNet.xpr.zip). 

Please load the archived project and update the HLS accelerator path.  

# Test Bench usage:

We provide some test bench data for verfication of a single layer

for conv 1x1:

	D = 32;        
	IC = 64;
	OC = 16;
	skip1 = 0; 
	skip3 = 1; 
	stride_2 = 0;
	deform = 0;
	readfile<FM_W, PA_0>(fmap_in, D*D*IC, "D_32_IC_64_OC_16_k_1_PA_16_PE_16input.txt");
	readfilepack<S_W, 1>(quant, OC*2, "D_32_IC_64_OC_16_k_1_PA_16_PE_16quant.txt");
	readfile<W_W, PA_0>(wtemp1,  IC * OC, "D_32_IC_64_OC_16_k_1_PA_16_PE_16weight1.txt");

for conv 3x3:

	D = 32;       
	IC = 64;
	OC = 64;
	skip1 = 1; 
	skip3 = 0; 
	stride_2 = 0;
	deform = 0;
	readfile<FM_W, PA_0>(fmap_in, D * D * IC, "D_32_C_64_k_3_PA_16_PE_16input.txt");
	readfilepack<S_W, 1>(quant, OC * 2, "D_32_C_64_k_3_PA_16_PE_16quant.txt");
	readfile<W_W, PA_0>(wtemp3,  IC * 9, "D_32_C_64_k_3_PA_16_PE_16weight3.txt");