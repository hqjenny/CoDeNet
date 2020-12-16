#include "ap_int.h"
#include "hls_stream.h"

using namespace std;
#define IN_W 8

#define OUT_W 8

typedef ap_uint<IN_W> T_IN;
typedef ap_uint<OUT_W> T_OUT;

#define FM_W 8
#define W_W 8
#define S_W 16
#define B_W 16
#define SHIFT_W0 16
typedef ap_uint<W_W> T_W;
#define SUM_W 18

typedef ap_uint<SUM_W> T_SUM;
 
#define PE_3 16
//#define PA_0 16
#define PA_0 16
#define PE_0 16

#define MAX_D 512
#define MAX_IC 512
#define MAX_OC 512
#define MAX_C 512
//#define PA_3 32
//#define PE_3 16
#define LINE_BUFF_MAX 3
#define KERNEL  3
#define PADDING_MAX 1
