#include "ap_int.h"
#include "hls_stream.h"

using namespace std;
#define IN_W 8
#define OUT_W 8

#define FM_W 8  //8 bits feature map
#define W_W 4   //4 bits weights
#define S_W 16  //16 bits scale
#define B_W 16  //16 bits bias
#define SHIFT_W0 16 //16 bits shift

#define SUM_W 24 //18 bits partial sum

typedef ap_uint<SUM_W> T_SUM;

#define PE_3 16
#define PA_0 16
#define PE_0 16

#define MAX_D 512  //MAX image size
#define MAX_IC 512 // MAX input channel
#define MAX_OC 1024 //MAX output channel
#define MAX_C 1024  //MAX 3x3 channel
#define MAX_WC 512 * 32 // MAX linebuffer width


