#ifndef __OPENCL_VERSION__
#include "header.cl"
#endif

__kernel void TEMPLATE(conv_layer_spatial_phony,Dtype)(KERNEL_ARG_DTYPE arg) {
  Dtype out = arg;
}

#ifdef FUSED_CONV_RELU
#define ACTIVATION_RELU_FUNCTION(x) ((Dtype)(x) > 0 ? (Dtype)(x) : ((Dtype)(x) * (Dtype)(negative_slope)))
#define NEGATIVE_SLOPE_ARG KERNEL_ARG_DTYPE negative_slope,
#else
#define ACTIVATION_RELU_FUNCTION(x) (x)
#define NEGATIVE_SLOPE_ARG
#endif

#ifdef FUSED_CONV_ELTWISE
#define ACTIVATION_FUNCTION(_dst_, _offset_, _data_) do { (_dst_)[(_offset_)] = ACTIVATION_RELU_FUNCTION(eltwise_data[(_offset_)] + (_data_));} while(0)
#define ELTWISE_DATA_ARG __global Dtype* eltwise_data,
#else
#define ACTIVATION_FUNCTION(_dst_, _offset_, _data_) do { (_dst_)[(_offset_)] = ACTIVATION_RELU_FUNCTION(_data_);} while(0)
#define ELTWISE_DATA_ARG
#endif

#define __CAT(x, y) x##y
#define CAT(x, y) __CAT(x, y)
#define LOOP0(VAR, STMT)
#define LOOP1(VAR, STMT) (STMT); (VAR)++;
#define LOOP2(VAR, STMT) LOOP1(VAR, STMT); (STMT); (VAR)++;
#define LOOP3(VAR, STMT) LOOP2(VAR, STMT); (STMT); (VAR)++;
#define LOOP4(VAR, STMT) LOOP3(VAR, STMT); (STMT); (VAR)++;
#define LOOP5(VAR, STMT) LOOP4(VAR, STMT); (STMT); (VAR)++;
#define LOOP6(VAR, STMT) LOOP5(VAR, STMT); (STMT); (VAR)++;
#define LOOP7(VAR, STMT) LOOP6(VAR, STMT); (STMT); (VAR)++;
#define LOOP8(VAR, STMT) LOOP7(VAR, STMT); (STMT); (VAR)++;
#define LOOP9(VAR, STMT) LOOP8(VAR, STMT); (STMT); (VAR)++;
#define LOOP10(VAR, STMT) LOOP9(VAR, STMT); (STMT); (VAR)++;
#define LOOP11(VAR, STMT) LOOP10(VAR, STMT); (STMT); (VAR)++;
#define LOOP12(VAR, STMT) LOOP11(VAR, STMT); (STMT); (VAR)++;
#define LOOP13(VAR, STMT) LOOP12(VAR, STMT); (STMT); (VAR)++;
#define LOOP14(VAR, STMT) LOOP13(VAR, STMT); (STMT); (VAR)++;
#define LOOP15(VAR, STMT) LOOP14(VAR, STMT); (STMT); (VAR)++;
#define LOOP16(VAR, STMT) LOOP15(VAR, STMT); (STMT); (VAR)++;
#define LOOP(N, VAR, STMT) CAT(LOOP, N)((VAR), (STMT))

#ifdef MULTI
__kernel void CFMultiNoPadding(
    ELTWISE_DATA_ARG
    NEGATIVE_SLOPE_ARG
    __global Dtype* image_data,
    int_tp image_offset,
    __global Dtype* kernel_data, int_tp kernel_offset,
    __global Dtype* bias,const int_tp bias_offset,
    __global Dtype* convolved_image,const int_tp convolved_image_offset,
    const ushort input_width,
    const ushort input_height,
    const ushort output_width,
    const ushort output_height,
    const ushort pad_w,
    const ushort pad_h) {

  const int_tp outputX = get_global_id(0);
  const int_tp outputY = get_global_id(1);
  const int_tp kernelNum = get_global_id(2)*ZPAR;
  if(outputX < output_width && outputY < output_height)
  {
    Dtype sum[ZPAR];
    for(int_tp kern =0; kern < ZPAR; kern++)
    {
      sum[kern] = 0.0f;
    }

    const int_tp org_y = outputY * STRIDE_H - pad_h;
    const int_tp org_x = outputX * STRIDE_W - pad_w;
    const int_tp currentKernelOffset = kernel_offset + kernelNum*KERNEL_H*KERNEL_W*CHANNELS;
    const int_tp biasIndex=bias_offset + kernelNum;
    const int_tp local_image_offset = org_y*input_width + org_x;
    const int_tp imageSize = input_width*input_height;

    __global Dtype* image_dataPtrFloat = (image_data + (image_offset + local_image_offset));
    __global Dtype* kernel_dataPtrFloat = (kernel_data + (currentKernelOffset));

    for(int_tp c = 0; c < CHANNELS; c++)
    {
      for(int_tp y = 0; y < KERNEL_H; y++)
      {
        for(int_tp x = 0; x < KERNEL_W; x++)
        {
          if(!(org_y + y * DILATION_Y >= 0 && org_y + y * DILATION_Y < input_height && org_x + x * DILATION_X >= 0 && org_x + x * DILATION_X < input_width))
          {
            continue;
          }
          for(int_tp kern =0; kern < ZPAR; kern++)
          {
            sum[kern] += image_dataPtrFloat[x * DILATION_X] * kernel_dataPtrFloat[kern*KERNEL_H*KERNEL_W*CHANNELS + x];
          }
        }
        image_dataPtrFloat += input_width * DILATION_Y;
        kernel_dataPtrFloat += KERNEL_W;
      }
      image_dataPtrFloat += imageSize - input_width*KERNEL_H*DILATION_Y;
    }

    if(APPLY_BIAS == 1)
    {
      for(int_tp kern = 0; kern < ZPAR; kern++)
      {
        if(kernelNum+kern < OUTPUT_Z)
        {
            int_tp offset = convolved_image_offset + (kernelNum+kern)*output_height*output_width + outputY*output_width + outputX;
            ACTIVATION_FUNCTION(convolved_image, offset, sum[kern] + bias[biasIndex +kern]);
        }
      }
    }
    else
    {
        for(int_tp kern = 0; kern < ZPAR; kern++)
        {
            if(kernelNum+kern < OUTPUT_Z)
            {
                int_tp offset = convolved_image_offset + (kernelNum+kern)*output_height*output_width + outputY*output_width + outputX;
                ACTIVATION_FUNCTION(convolved_image, offset, sum[kern]);
            }
        }
    }
  }
}

#endif

#if defined(convolve_simd) || defined(Conv_Interleaved)
#if TYPE == TYPE_HALF
#define INT_TYPE ushort
#define INT_TYPE2 ushort2
#define INT_TYPE4 ushort4
#define INT_TYPE8 ushort8
#define SUB_GROUP_BLOCK_READ2 intel_sub_group_block_read_us2
#define SUB_GROUP_BLOCK_READ4 intel_sub_group_block_read_us4
#define SUB_GROUP_BLOCK_READ8 intel_sub_group_block_read_us8
#define SUB_GROUP_BLOCK_READ intel_sub_group_block_read_us
#else
#define INT_TYPE uint
#define INT_TYPE2 uint2
#define INT_TYPE4 uint4
#define INT_TYPE8 uint8
#define SUB_GROUP_BLOCK_READ2 intel_sub_group_block_read2
#define SUB_GROUP_BLOCK_READ4 intel_sub_group_block_read4
#define SUB_GROUP_BLOCK_READ8 intel_sub_group_block_read8
#define SUB_GROUP_BLOCK_READ intel_sub_group_block_read
#endif
#endif

//Begin IDLF kernels below here
#ifdef IDLF

#define OUT_BLOCK_SIZE (OUT_BLOCK_WIDTH*OUT_BLOCK_HEIGHT)

// Each work-item computes a OUT_BLOCK_WIDTH * OUT_BLOCK_HEIGHT region of one output map.
// Each work-group (which will be mapped to 1 SIMD16/SIMD8 EU thread) will compute 16/8 different feature maps, but each feature map is for the same region of the imput image.
// NDRange:  (output_width+pad)/ OUT_BLOCK_WIDTH, (output_height+pad)/OUT_BLOCK_HEIGHT, NUM_FILTERS/OUT_BLOCK_DEPTH

// NOTE: for beignet this reqd_work_group_size does not guarantee that SIMD16 mode will be used, the compiler could choose to use two SIMD8 threads, and if that happens the code will break.
#ifndef __BEIGNET__
__attribute__((reqd_work_group_size(1, 1, SIMD_SIZE)))
#endif
__kernel void
convolve_simd(
    ELTWISE_DATA_ARG
    NEGATIVE_SLOPE_ARG
    __global Dtype* inputs_base,
    filter_qualifier Dtype* weights_base,
    __global Dtype* biases_base,
    __global Dtype* outputs_base,
    const ushort input_width,
    const ushort input_height,
    const ushort output_width,
    const ushort output_height)
{
  __global Dtype* outputs = outputs_base;
  __global Dtype* inputs = inputs_base;
  filter_qualifier Dtype* weights = weights_base;
  __global Dtype* biases = biases_base;

  uint_tp oc = get_global_id(0) * OUT_BLOCK_WIDTH;  // oc = Output Column
  uint_tp or = get_global_id(1) * OUT_BLOCK_HEIGHT;// or = Output Row
  uint_tp fm = get_global_id(2);// fm = Feature Map = od = Output Depth
  uint_tp fmg = get_group_id(2);
  uint_tp lid = get_local_id(2);

  Dtype out[OUT_BLOCK_SIZE];

  int_tp in_addr;

  // find weights adress of given neuron (lid is index)
  uint_tp weight_addr = (fmg % (ALIGNED_NUM_FILTERS/SIMD_SIZE)) * INPUT_DEPTH * KERNEL_WIDTH * KERNEL_HEIGHT * SIMD_SIZE + lid;

  for(int_tp i=0;i<OUT_BLOCK_SIZE;i++) {
    out[i]=0.0f;
  }

  uint_tp num_in_batch = ( fm ) / ALIGNED_NUM_FILTERS;

  uint_tp input_batch_offset = num_in_batch * input_height * input_width * TOTAL_INPUT_DEPTH_SIZE;

  int curr_local_y = ( lid / ( TILE_X / 4 ) );
  int curr_local_x = ( lid % ( TILE_X / 4 ) ) * 4;
  int curr_y = or * STRIDEY + INPUT_START_Y + curr_local_y;
  int curr_x = oc * STRIDEX + INPUT_START_X + curr_local_x;
#if INPUT_PAD_W != 0 || INPUT_PAD_H != 0
  int saved_y = curr_y;
#endif
  in_addr = input_batch_offset + INPUT_START_Z * input_height * input_width
            +  (curr_y - INPUT_PAD_H) * input_width             // y tile offset
            +   curr_x - INPUT_PAD_W;                        // x tile offset
  union {
    Dtype4 in_vec[INVEC_SIZE];
    Dtype in_array[INVEC_SIZE * 4];
  } in_buf;

  for(int_tp kd = 0; kd < INPUT_DEPTH; kd++)
  {
    int_tp in_offset = in_addr;
    int_tp reg = 0;
    LOOP(INVEC_SIZE, reg,
      {
        if (curr_local_y + reg * TILE_Y_STRIDE < TILE_Y || INVEC_SIZE * TILE_Y_STRIDE == TILE_Y || reg < INVEC_SIZE - 1) {
#if INPUT_PAD_W != 0 || INPUT_PAD_H != 0
        if (curr_y >= INPUT_PAD_H && curr_y < input_height + INPUT_PAD_H && curr_x + 3 >= INPUT_PAD_W && curr_x < input_width + INPUT_PAD_W) {
          if (curr_x < INPUT_PAD_W) {
            in_buf.in_vec[reg].s0 = 0;
            if (curr_x + 1 >= INPUT_PAD_W)
              in_buf.in_vec[reg].s1 = *(inputs + in_offset + 1);
            else
              in_buf.in_vec[reg].s1 = 0;
            if (curr_x + 2 >= INPUT_PAD_W)
              in_buf.in_vec[reg].s2 = *(inputs + in_offset + 2);
            else
              in_buf.in_vec[reg].s2 = 0;
            in_buf.in_vec[reg].s3 = *(inputs + in_offset + 3);
          } else {
            in_buf.in_vec[reg] = vload4(0, (inputs + in_offset)); // read 16 elements
            if (curr_x + 1 >= input_width + INPUT_PAD_W)
              in_buf.in_vec[reg].s1 = 0;
            if (curr_x + 2 >= input_width + INPUT_PAD_W)
              in_buf.in_vec[reg].s2 = 0;
            if (curr_x + 3 >= input_width + INPUT_PAD_W)
              in_buf.in_vec[reg].s3 = 0;
          }
        } else {
          in_buf.in_vec[reg] = 0;
        }
        curr_y += TILE_Y_STRIDE;
#else
        in_buf.in_vec[reg] = vload4(0, (inputs + in_offset)); // read 16 elements
#endif
        }
        in_offset += input_width * TILE_Y_STRIDE;
      });
    in_addr += input_height * input_width;
#if INPUT_PAD_W != 0 || INPUT_PAD_H != 0
    curr_y = saved_y;
#endif

#if KERNEL_WIDTH * KERNEL_HEIGHT != 1
#define WEIGHT_PREF 8
#else
#define WEIGHT_PREF 1
#endif
    union {
      Dtype w[WEIGHT_PREF];
#if KERNEL_WIDTH * KERNEL_HEIGHT != 1
      INT_TYPE8 ui8;
#endif
    } weight_buf;
    int_tp w_idx=0;

    uint_tp orig_weight_addr = weight_addr;
#if KERNEL_WIDTH * KERNEL_HEIGHT != 1
    weight_buf.ui8 = SUB_GROUP_BLOCK_READ8((__global INT_TYPE *)&weights[weight_addr]);
    weight_addr += SIMD_SIZE * WEIGHT_PREF;
#else
    weight_buf.w[0] = as_Dtype(SUB_GROUP_BLOCK_READ((__global INT_TYPE *)&weights[weight_addr]));
    weight_addr += SIMD_SIZE * 1;
#endif

#define BLOCK_IN(n) sub_group_broadcast( in_buf.in_array[((n)%4) + ((n) / (TILE_Y_STRIDE * TILE_X)) * 4], (((n) % (TILE_Y_STRIDE * TILE_X))/4))

    int_tp kr = 0;  // kr = Kernel Row
    LOOP(KERNEL_HEIGHT, kr,// LOOP is a macro that unrolls the loop.
        {
          int_tp kc = 0;  // kc = Kernel Column
          LOOP(KERNEL_WIDTH, kc,
              {
                for(int_tp br=0; br < OUT_BLOCK_HEIGHT; br++) {
                  for(int_tp bc=0; bc < OUT_BLOCK_WIDTH; bc++) {
                    Dtype input = BLOCK_IN((br * STRIDEY + kr * DILATION_Y) * TILE_X + bc * STRIDEX + kc * DILATION_X);
                    out[br * OUT_BLOCK_WIDTH + bc] = mad(weight_buf.w[w_idx % WEIGHT_PREF], input, out[br * OUT_BLOCK_WIDTH + bc]);
                  }
                }
#if KERNEL_WIDTH * KERNEL_HEIGHT > WEIGHT_PREF
                // We assume KERNEL_W is equal to KERNEL_H here.
                if ((w_idx + 1) % WEIGHT_PREF == 0
                #if KERNEL_WIDTH * KERNEL_HEIGHT % 8 != 0
                && ((w_idx + 1) <= (KERNEL_WIDTH * KERNEL_HEIGHT - WEIGHT_PREF))
                #endif
                    ) {
                  weight_buf.ui8 = SUB_GROUP_BLOCK_READ8((__global INT_TYPE *)&weights[weight_addr]);
                  weight_addr += SIMD_SIZE * WEIGHT_PREF;  // weights must be stored in just the right SIMD swizzled format for this to work, see host code for details.
                }
              #if KERNEL_WIDTH*KERNEL_HEIGHT % 8 == 0
                // need to do nothing
              #else
                else if ((w_idx + 1) %  WEIGHT_PREF == 0 && ((w_idx + 1) > (KERNEL_WIDTH * KERNEL_HEIGHT - WEIGHT_PREF)))
                #if KERNEL_WIDTH * KERNEL_HEIGHT % 8 == 1
                  weight_buf.w[0] = weights[weight_addr];
                #elif KERNEL_WIDTH * KERNEL_HEIGHT % 8 == 2
                  weight_buf.ui8.s01 = SUB_GROUP_BLOCK_READ2((__global INT_TYPE *)&weights[weight_addr]);
                #elif KERNEL_WIDTH * KERNEL_HEIGHT % 8 <= 4
                  weight_buf.ui8.s0123 = SUB_GROUP_BLOCK_READ4((__global INT_TYPE *)&weights[weight_addr]);
                #else
                  weight_buf.ui8 = SUB_GROUP_BLOCK_READ8((__global INT_TYPE *)&weights[weight_addr]);
                #endif
              #endif
#endif
                ++w_idx;
              });
        });
    weight_addr = orig_weight_addr + KERNEL_WIDTH * KERNEL_HEIGHT * SIMD_SIZE;

  }
  // dead code to work around possible compiler bug.
  if (ALIGNED_NUM_FILTERS != NUM_FILTERS && fm > 0xfffffffeul) {
    outputs[0] = BLOCK_IN(fm % SIMD_SIZE);
  }
  fm = fm % ALIGNED_NUM_FILTERS;

  if ((ALIGNED_NUM_FILTERS == NUM_FILTERS || fm < NUM_FILTERS)) {
  uint_tp out_addr = OUT_BUFF_OFFSET + ( num_in_batch * TOTAL_OUTPUT_DEPTH + fm ) * output_width * output_height;
  out_addr += or * output_width + oc;
  // we need this address calculation for biases because we support views and batching
  Dtype bias = biases[fm];
    for(uint_tp r = 0; r < OUT_BLOCK_HEIGHT; r++) {
      if (r + or >= output_height) break;
      for(uint_tp c = 0; c < OUT_BLOCK_WIDTH; c++) {
        if (c + oc >= output_width) break;
        // this does a scattered write to SIMD_SIZE different feature maps, so that data within one map is contiguous, thus ready for input to next layer.
          ACTIVATION_FUNCTION(outputs, out_addr + r * output_width + c, bias + out[r * OUT_BLOCK_WIDTH + c]);
      }
    }
  }
}
#endif

/*******************************************************************************
Copyright © 2016, Intel Corporation

    Permission is hereby granted, free of charge, to any person obtaining a
    copy of this software and associated documentation files (the "Software"),
    to deal in the Software without restriction, including without limitation
    the rights to use, copy, modify, merge, publish, distribute, sublicense,
    and/or sell copies of the Software, and to permit persons to whom the
    Software is furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in
    all copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
    THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
    FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
    DEALINGS IN THE SOFTWARE.
******************************************************************************/
#ifdef Conv_Interleaved
typedef struct float1 { float s0; } float1;
typedef struct float5 { float s0; float s1; float s2; float s3; float s4; } float5;
typedef struct float6 { float s0; float s1; float s2; float s3; float s4; float s5; } float6;
typedef struct float7 { float s0; float s1; float s2; float s3; float s4; float s5; float s6; } float7;
typedef struct float9 { float s0; float s1; float s2; float s3; float s4; float s5; float s6; float s7; float s8; } float9;
typedef struct float10 { float s0; float s1; float s2; float s3; float s4; float s5;
                         float s6; float s7; float s8; float s9;} float10;
typedef struct float11 { float s0; float s1; float s2; float s3; float s4; float s5;
                         float s6; float s7; float s8; float s9; float sa;} float11;
typedef struct float12 { float s0; float s1; float s2; float s3; float s4; float s5;
                         float s6; float s7; float s8; float s9; float sa; float sb; } float12;
typedef struct float13 { float s0; float s1; float s2; float s3; float s4; float s5;
                         float s6; float s7; float s8; float s9; float sa; float sb; float sc;} float13;
typedef struct float14 { float s0; float s1; float s2; float s3; float s4; float s5;
                         float s6; float s7; float s8; float s9; float sa; float sb; float sc; float sd; } float14;
typedef struct float15 { float s0; float s1; float s2; float s3; float s4; float s5;
                         float s6; float s7; float s8; float s9; float sa; float sb; float sc; float sd; float se; } float15;
typedef struct float0 { float s0; } float0; //never used but makes compiler happy.

typedef struct half1 { half s0; } half1;
typedef struct half5 { half s0; half s1; half s2; half s3; half s4; } half5;
typedef struct half6 { half s0; half s1; half s2; half s3; half s4; half s5; } half6;
typedef struct half7 { half s0; half s1; half s2; half s3; half s4; half s5; half s6; } half7;
typedef struct half9 { half s0; half s1; half s2; half s3; half s4; half s5; half s6; half s7; half s8; } half9;
typedef struct half10 { half s0; half s1; half s2; half s3; half s4; half s5;
                        half s6; half s7; half s8; half s9; } half10;
typedef struct half11 { half s0; half s1; half s2; half s3; half s4; half s5;
                        half s6; half s7; half s8; half s9; half sa; } half11;
typedef struct half12 { half s0; half s1; half s2; half s3; half s4; half s5;
                        half s6; half s7; half s8; half s9; half sa; half sb; } half12;
typedef struct half13 { half s0; half s1; half s2; half s3; half s4; half s5;
                        half s6; half s7; half s8; half s9; half sa; half sb; half sc; } half13;
typedef struct half14 { half s0; half s1; half s2; half s3; half s4; half s5;
                        half s6; half s7; half s8; half s9; half sa; half sb; half sc; half sd; } half14;
typedef struct half15 { half s0; half s1; half s2; half s3; half s4; half s5;
                        half s6; half s7; half s8; half s9; half sa; half sb; half sc; half sd; half se; } half15;
typedef struct half0 { half s0; } half0; //never used but makes compiler happy.

#define OUT_PITCH_X output_width
#define ROW_PITCH input_width


#define GEMM_LIKE_KERNEL_ARGS     \
    ELTWISE_DATA_ARG              \
    NEGATIVE_SLOPE_ARG            \
    const __global Dtype *src0,   \
    const __global Dtype *src1,   \
    const __global Dtype *biases, \
    __global Dtype *dst,          \
    const ushort input_width,     \
    const ushort input_height,    \
    const ushort output_width,    \
    const ushort output_height,   \
    const int_tp out_pitch_y,     \
    const int_tp out_pitch_z,     \
    const int_tp aligned_input_size, \
    const int_tp slice_pitch

#endif



#ifdef GEMM_LIKE_CONV_32_1
//////////////////////////////////////////////////////////////////////////////
// Conv_Interleaved_32_1_flex
//
// Convolution: each workitem computes 1 patch x 32 filters worth of output 
// data.  Kernel's inner loop works on a single tile consisting of one
// row from each patch and the filter data corresponding to that row.  Filter
// matrix is interleaved to reduce GRF bank conflicts.  Patches are walked
// by rows and then by slices.  Relies on sub_group extension for block
// reads and SIMD broadcast.  Allows flexible sizing of TILE width (TILE_N) 
// by dynamically selecting one of two code paths: one uses TILE_N = 32 and 
// the other uses TILE_N = 8, 16, or 24.
#define TILE_M          1
#define TILE_K          KERNEL_WIDTH
#define TILE_N          32

#ifndef __BEIGNET__
__attribute__((intel_reqd_sub_group_size(8)))
#endif
__kernel void Conv_Interleaved(GEMM_LIKE_KERNEL_ARGS)
{
    const int group_x = get_group_id(0);
    const int group_y = get_group_id(1);
    const int global_x = get_global_id(0);
    const int global_y = get_global_id(1);
    const int global_z = get_global_id(2);
    int interleaved_y;
    int kernel_y;
    int kernel_idx;
    
#define DOT_PRODUCT_8( _result, _rowA, colB )    \
    {   \
        _result.s0 = mad( _rowA, sub_group_broadcast( colB, 0 ), _result.s0 );  \
        _result.s1 = mad( _rowA, sub_group_broadcast( colB, 1 ), _result.s1 );  \
        _result.s2 = mad( _rowA, sub_group_broadcast( colB, 2 ), _result.s2 );  \
        _result.s3 = mad( _rowA, sub_group_broadcast( colB, 3 ), _result.s3 );  \
        _result.s4 = mad( _rowA, sub_group_broadcast( colB, 4 ), _result.s4 );  \
        _result.s5 = mad( _rowA, sub_group_broadcast( colB, 5 ), _result.s5 );  \
        _result.s6 = mad( _rowA, sub_group_broadcast( colB, 6 ), _result.s6 );  \
        _result.s7 = mad( _rowA, sub_group_broadcast( colB, 7 ), _result.s7 );  \
    }
    typedef CAT( Dtype, KERNEL_WIDTH ) Dtype_t;

    // True for all threads if filter_width is multiple of TILE_N
    // else, true for all but right-most column of threads.
    if( TILE_N_LAST == 0 || global_x < WIDTH1 / TILE_N ) 
    {
        // Result ctile (*dst) is M rows x N columns
        // LWG size is 1x8.  Thus each thread calculates 8*M rows x N cols of ctile.
        Dtype8  blockC00 = 0.f;
        Dtype8  blockC10 = 0.f;
        Dtype8  blockC20 = 0.f;
        Dtype8  blockC30 = 0.f;

        // Src0 (patch input) is directly used as atile.
        // Each work item points to the start of a different patch.
        // atile is M rows x K columns.
        int curr_x = ( global_y % output_width ) * STRIDE_X;
        int curr_y = ( global_y / output_width ) * STRIDE_Y;
#if INPUT_PAD_H != 0 || INPUT_PAD_W != 0 || DILATION_X != 1 || DILATION_Y != 1
        int saved_y = curr_y;
#endif
        const __global Dtype *src0_read = src0
          + aligned_input_size * global_z                            // batch offset
          + (curr_y - INPUT_PAD_H) * ROW_PITCH      // y offset
          + (curr_x - INPUT_PAD_W);                 // x offset

        // Src1 (filter) is directly used as btile.
        // It starts at the top of src1 and walks down.
        // btile is K rows x N columns.
        const __global Dtype *src1_read = src1 + ( global_x * TILE_N  * 2);

        // Walk DOWN src0 (patch 0, 1, 2, ...) and DOWN src1.
        // Inner loop loads and FMADs one row (KERNEL_WIDTH) of each input patch 
        // and KERNEL_WIDTH/2 rows of interleaved filter.
        int patch_depth = 0;
        do
        {
            int patch_row = 0;
#if INPUT_PAD_H != 0 || INPUT_PAD_W != 0 || DILATION_X != 1 || DILATION_Y != 1
            curr_y = saved_y;
#endif

            do
            {
                // Load atile and btile.
                // Kernel data is partially interleaved.  Every 2 rows are interleaved at Dtype8 granularity.
                // The exception is that if KERNEL_WIDTH is odd the last row is not interleaved.  The non 
                // interleaved row is padded with zero to ensure same size as interleaved rows. This
                // interleaving is done to ensure 0% GDR bank conflicts.  For example, this is how the
                // kernel data would be arranged before/after interleaving for KERNEL_WIDTH=3.
                // (0, 0) (8, 0) (16, 0) (24, 0) ...       (0, 0) (0, 1) (8, 0) (0, 1) (16, 0) (0, 1) (24, 0) ..
                // (0, 1) (8, 1) (16, 1) (24, 1) ... =>    (0, 2) (8, 2) (16, 2) (24, 2) ...
                // (0, 2) (8, 2) (16, 2) (24, 2) ...       ...
                // ...
                const bool kernel_width_is_odd = KERNEL_WIDTH % 2 == 1;

#if INPUT_PAD_W == 0 && INPUT_PAD_H == 0 && DILATION_X == 1 && DILATION_Y == 1
                Dtype_t blockA00 = ( (const __global Dtype_t*)src0_read )[  0  ];
                Dtype*  pblockA00 = (Dtype*)(&blockA00);
#else
                Dtype_t blockA00;
                Dtype*  pblockA00 = (Dtype*)(&blockA00);
                int pos = 0;
                LOOP(KERNEL_WIDTH, pos,
                {
                  if (curr_y >= INPUT_PAD_H && curr_y < input_height + INPUT_PAD_H && curr_x + pos * DILATION_X >= INPUT_PAD_W && curr_x + pos * DILATION_X < input_width + INPUT_PAD_W)
                    pblockA00[pos] = src0_read[pos * DILATION_X];
                  else
                    pblockA00[pos] = 0;
                })
                curr_y += DILATION_Y;
#endif
                src0_read += (ROW_PITCH * DILATION_Y);

                Dtype blockB00[KERNEL_WIDTH*4];
                Dtype8* p8BlockB00 = (Dtype8*)blockB00;
                Dtype4* p4BlockB00 = (Dtype4*)blockB00;
                Dtype*  pBlockB00 =  (Dtype* )blockB00;

                interleaved_y = 0;
                LOOP(KERNEL_WIDTH_DIV2, interleaved_y, 
                { 
                    p8BlockB00[interleaved_y] = as_Dtype8( SUB_GROUP_BLOCK_READ8( (const __global INT_TYPE *)src1_read ) );
                    src1_read += WIDTH1 * 2;
                } )
                if ( kernel_width_is_odd )
                {
                    p4BlockB00[KERNEL_WIDTH - 1] = as_Dtype4( SUB_GROUP_BLOCK_READ4( (const __global INT_TYPE *)src1_read ) );
                    src1_read += WIDTH1 * 2;
                }

                // Perform MADs
                kernel_idx = 0;
                interleaved_y = 0;
                LOOP(KERNEL_WIDTH_DIV2, interleaved_y, 
                {
                    kernel_y = interleaved_y * 2;
                    DOT_PRODUCT_8( blockC00, pblockA00[kernel_y    ], pBlockB00[kernel_idx] ); kernel_idx++;
                    DOT_PRODUCT_8( blockC00, pblockA00[kernel_y + 1], pBlockB00[kernel_idx] ); kernel_idx++;
                    DOT_PRODUCT_8( blockC10, pblockA00[kernel_y    ], pBlockB00[kernel_idx] ); kernel_idx++;
                    DOT_PRODUCT_8( blockC10, pblockA00[kernel_y + 1], pBlockB00[kernel_idx] ); kernel_idx++;
                    DOT_PRODUCT_8( blockC20, pblockA00[kernel_y    ], pBlockB00[kernel_idx] ); kernel_idx++;
                    DOT_PRODUCT_8( blockC20, pblockA00[kernel_y + 1], pBlockB00[kernel_idx] ); kernel_idx++;
                    DOT_PRODUCT_8( blockC30, pblockA00[kernel_y    ], pBlockB00[kernel_idx] ); kernel_idx++;
                    DOT_PRODUCT_8( blockC30, pblockA00[kernel_y + 1], pBlockB00[kernel_idx] ); kernel_idx++;
                } )
                    kernel_y = interleaved_y * 2;
                if ( kernel_width_is_odd )
                {
                    DOT_PRODUCT_8( blockC00, pblockA00[kernel_y], pBlockB00[kernel_idx] ); kernel_idx++;
                    DOT_PRODUCT_8( blockC10, pblockA00[kernel_y], pBlockB00[kernel_idx] ); kernel_idx++;
                    DOT_PRODUCT_8( blockC20, pblockA00[kernel_y], pBlockB00[kernel_idx] ); kernel_idx++;
                    DOT_PRODUCT_8( blockC30, pblockA00[kernel_y], pBlockB00[kernel_idx] ); kernel_idx++;
                }
            }

            //while( ++patch_row < 1 ); //debug
            while( ++patch_row < KERNEL_HEIGHT );

            src0_read += slice_pitch - ( KERNEL_HEIGHT * ROW_PITCH * DILATION_Y); // reset to start of next slice of patch
        } 
        //while ( ++patch_depth < 1 ); //debug
        while ( ++patch_depth < INPUT_DEPTH );
    
        // Dst resembles a cube of width x height x (output channel * batches).  Each tile writes: 
        // (SIMD * TILE_M) x 1 x TILE_N.  Partial writes most likely generated if padding used.
        int_tp out_offset = global_z * out_pitch_z                                                   // batch offset
         + ( group_x * TILE_N ) * out_pitch_y                                       // channel offset
         + ( ( global_y * TILE_M ) / output_width + OUT_PADDING_HEIGHT) * OUT_PITCH_X  // y offset
         + ( ( global_y * TILE_M ) % output_width ) + OUT_PADDING_LEFT;               // x offset

        __global Dtype *out = dst + out_offset;
        Dtype bias[4];
        Dtype4 *bias_vec;
        bias_vec = (Dtype4*)bias;
        *bias_vec = as_Dtype4(SUB_GROUP_BLOCK_READ4((__global INT_TYPE *)biases + group_x * TILE_N));

        if (global_y * TILE_M < output_width * output_height )
        {
            for (int i = 0; i < 8; i++)
            {
                ACTIVATION_FUNCTION(dst, out_offset + ( 0 + i ) * out_pitch_y, blockC00[i] + intel_sub_group_shuffle(bias[0], i));
                ACTIVATION_FUNCTION(dst, out_offset + ( 8 + i ) * out_pitch_y, blockC10[i] + intel_sub_group_shuffle(bias[1], i));
                ACTIVATION_FUNCTION(dst, out_offset + ( 16 + i ) * out_pitch_y, blockC20[i] + intel_sub_group_shuffle(bias[2], i));
                ACTIVATION_FUNCTION(dst, out_offset + ( 24 + i ) * out_pitch_y, blockC30[i] + intel_sub_group_shuffle(bias[3], i));
            }
        }
    }
#if TILE_N_LAST > 0
    else
    {
        
        // Result ctile (*dst) is M rows x N columns
        // LWG size is 1x8.  Thus each thread calculates 8*M rows x N cols of ctile.
        int i = 0;
        Dtype8  blockC[TILE_N_LAST_DIV8];
        LOOP(TILE_N_LAST_DIV8, i,
        {
            blockC[i] = 0.f;
        } )
        
        // Src0 (patch input) is directly used as atile.
        // Each work item points to the start of a different patch.
        // atile is M rows x K columns.
        int curr_x = ( global_y % output_width ) * STRIDE_X;
        int curr_y = ( global_y / output_width ) * STRIDE_Y;
#if INPUT_PAD_H != 0 || INPUT_PAD_W != 0 || DILATION_X != 1 || DILATION_Y != 1
        int saved_y = curr_y;
#endif
        const __global Dtype *src0_read = src0
          + aligned_input_size * global_z                            // batch offset
          + (curr_y - INPUT_PAD_H) * ROW_PITCH      // y offset
          + (curr_x - INPUT_PAD_W);                 // x offset

        // Src1 (filter) is directly used as btile.
        // It starts at the top of src1 and walks down.
        // btile is K rows x N columns.
        const __global Dtype *src1_read = src1 + ( global_x * TILE_N  * 2);

        // Walk DOWN src0 (patch 0, 1, 2, ...) and DOWN src1.
        // Inner loop loads and FMADs one row (KERNEL_WIDTH) of each input patch 
        // and KERNEL_WIDTH/2 rows of interleaved filter.
        int patch_depth = 0;
        do
        {
            int patch_row = 0;
#if INPUT_PAD_H != 0 || INPUT_PAD_W != 0 || DILATION_X != 1 || DILATION_Y != 1
            curr_y = saved_y;
#endif
            do
            {
                // Load atile and interleaved btile.
                const bool kernel_width_is_odd = KERNEL_WIDTH % 2 == 1;
#if INPUT_PAD_W == 0 && INPUT_PAD_H == 0 && DILATION_X == 1 && DILATION_Y == 1
                Dtype_t blockA00 = ( (const __global Dtype_t*)src0_read )[  0  ];
                Dtype*  pblockA00 = (Dtype*)(&blockA00);
#else
                Dtype_t blockA00;
                Dtype*  pblockA00 = (Dtype*)(&blockA00);
                int pos = 0;
                LOOP(KERNEL_WIDTH, pos,
                {
                  if (curr_y >= INPUT_PAD_H && curr_y < input_height + INPUT_PAD_H && curr_x + pos * DILATION_X >= INPUT_PAD_W && curr_x + pos * DILATION_X < input_width + INPUT_PAD_W)
                    pblockA00[pos] = src0_read[pos * DILATION_X];
                  else
                    pblockA00[pos] = 0;
                })
                curr_y += DILATION_Y;
#endif
                src0_read += (ROW_PITCH * DILATION_Y);
                Dtype blockB[KERNEL_WIDTH * TILE_N_LAST_DIV8];

                interleaved_y = 0;
                LOOP(KERNEL_WIDTH_DIV2, interleaved_y, 
                { 
#if TILE_N_LAST_DIV8 == 1
                    Dtype2* p2BlockB = (Dtype2* )blockB;
                    p2BlockB[interleaved_y] = as_Dtype2( SUB_GROUP_BLOCK_READ2( (const __global INT_TYPE*)src1_read ) );
#elif TILE_N_LAST_DIV8 == 2
                    Dtype4* p4BlockB = (Dtype4* )blockB;
                    p4BlockB[interleaved_y] = as_Dtype4( SUB_GROUP_BLOCK_READ4( (const __global INT_TYPE*)src1_read ) );
#elif TILE_N_LAST_DIV8 == 3
                    //TODO: broken.  No block_read6
                    Dtype6* p6BlockB = (Dtype6* )blockB;
                    (*((Dtype8*)(&p6BlockB[interleaved_y]))).s0123 = as_Dtype4( SUB_GROUP_BLOCK_READ4( (const __global INT_TYPE*)src1_read ) );
                    (*((Dtype8*)(&p6BlockB[interleaved_y]))).s45 = as_Dtype2( SUB_GROUP_BLOCK_READ2( (const __global INT_TYPE*)(src1_read + 4 * 8) ) );
#endif
                    src1_read += WIDTH1 * 2;
                } )
                if ( kernel_width_is_odd )
                {
#if TILE_N_LAST_DIV8 == 1
                    Dtype* pBlockB = (Dtype* )blockB;
                    pBlockB[KERNEL_WIDTH - 1] = as_Dtype( SUB_GROUP_BLOCK_READ( (const __global INT_TYPE*)src1_read ) );
#elif TILE_N_LAST_DIV8 == 2
                    Dtype2* p2BlockB = (Dtype2* )blockB;
                    p2BlockB[KERNEL_WIDTH - 1] = as_Dtype2( SUB_GROUP_BLOCK_READ2( (const __global INT_TYPE*)src1_read ) );
#elif TILE_N_LAST_DIV8 == 3
                    Dtype3* p3BlockB = (Dtype3* )blockB;
                    p3BlockB[KERNEL_WIDTH - 1].s01 = as_Dtype2( SUB_GROUP_BLOCK_READ2( (const __global INT_TYPE*)src1_read ) );
                    p3BlockB[KERNEL_WIDTH - 1].s2 = as_Dtype( SUB_GROUP_BLOCK_READ( (const __global INT_TYPE*) (src1_read + 2 * 8) ) );
#endif
                    src1_read += WIDTH1 * 2;
                }

                // Perform MADs
                Dtype* pBlockB = (Dtype*)blockB;
                kernel_idx = 0;
                interleaved_y = 0;
                LOOP(KERNEL_WIDTH_DIV2, interleaved_y, 
                {
                    kernel_y = interleaved_y * 2;
                    DOT_PRODUCT_8( blockC[0], pblockA00[kernel_y    ], pBlockB[kernel_idx] ); kernel_idx++;
                    DOT_PRODUCT_8( blockC[0], pblockA00[kernel_y + 1], pBlockB[kernel_idx] ); kernel_idx++;
#if TILE_N_LAST_DIV8 >= 2
                    DOT_PRODUCT_8( blockC[1], pblockA00[kernel_y    ], pBlockB[kernel_idx] ); kernel_idx++;
                    DOT_PRODUCT_8( blockC[1], pblockA00[kernel_y + 1], pBlockB[kernel_idx] ); kernel_idx++;
#if TILE_N_LAST_DIV8 >= 3
                    DOT_PRODUCT_8( blockC[2], pblockA00[kernel_y    ], pBlockB[kernel_idx] ); kernel_idx++;
                    DOT_PRODUCT_8( blockC[2], pblockA00[kernel_y + 1], pBlockB[kernel_idx] ); kernel_idx++;
#endif
#endif
                } )
                    kernel_y = interleaved_y * 2;
                if ( kernel_width_is_odd )
                {
                    DOT_PRODUCT_8( blockC[0], pblockA00[kernel_y], pBlockB[kernel_idx] ); kernel_idx++;
#if TILE_N_LAST_DIV8 >= 2
                    DOT_PRODUCT_8( blockC[1], pblockA00[kernel_y], pBlockB[kernel_idx] ); kernel_idx++;
#if TILE_N_LAST_DIV8 >= 3
                    DOT_PRODUCT_8( blockC[2], pblockA00[kernel_y], pBlockB[kernel_idx] ); kernel_idx++;
#endif
#endif
                }
            }

            //while( ++patch_row < 1 ); //debug
            while( ++patch_row < KERNEL_HEIGHT );

            src0_read += slice_pitch - ( KERNEL_HEIGHT * ROW_PITCH * DILATION_Y ); // reset to start of next slice of patch
        } 
        //while ( ++patch_depth < 1 );  //debug
        while ( ++patch_depth < INPUT_DEPTH );
    
        // Dst resembles a cube of width x height x (output channel * batches).  Each tile writes: 
        // (SIMD * TILE_M) x 1 x TILE_N.  Partial writes most likely generated if padding used.
        int_tp out_offset = global_z * out_pitch_z                                                   // batch offset
         + ( group_x * TILE_N ) * out_pitch_y                                       // channel offset
         + ( ( global_y * TILE_M ) / output_width + OUT_PADDING_HEIGHT) * OUT_PITCH_X  // y offset
         + ( ( global_y * TILE_M ) % output_width ) + OUT_PADDING_LEFT;               // x offset
        __global Dtype *out = dst + out_offset;
        Dtype bias[4];
        Dtype4 *bias_vec;
        bias_vec = (Dtype4*)bias;
        *bias_vec = as_Dtype4(SUB_GROUP_BLOCK_READ4((__global INT_TYPE *)biases + group_x * TILE_N));

        if (global_y * TILE_M < output_width * output_height )
        {
            for (int i = 0; i < 8; i++)
            {
                if ( TILE_N_LAST_DIV8 > 0 ) ACTIVATION_FUNCTION(dst, out_offset + ( 0+i) * out_pitch_y, blockC[0][i] + intel_sub_group_shuffle(bias[0], i));
                if ( TILE_N_LAST_DIV8 > 1 ) ACTIVATION_FUNCTION(dst, out_offset + ( 8+i) * out_pitch_y, blockC[1][i] + intel_sub_group_shuffle(bias[1], i));
                if ( TILE_N_LAST_DIV8 > 2 ) ACTIVATION_FUNCTION(dst, out_offset + (16+i) * out_pitch_y, blockC[2][i] + intel_sub_group_shuffle(bias[2], i));
                if ( TILE_N_LAST_DIV8 > 3 ) ACTIVATION_FUNCTION(dst, out_offset + (24+i) * out_pitch_y, blockC[3][i] + intel_sub_group_shuffle(bias[3], i));
            }
        }
    }
#endif
}
#endif
#ifdef GEMM_LIKE_CONV_32_2

//////////////////////////////////////////////////////////////////////////////
// Conv_Interleaved_32_2_flex
//
// Convolution: each workitem computes 1 patch x 32 filters worth of output 
// data.  Kernel's inner loop works on a single tile consisting of one
// row from each patch and the filter data corresponding to that row.  Filter
// matrix is interleaved to reduce GRF bank conflicts.  Patches are walked
// by rows and then by slices.  Relies on sub_group extension for block
// reads and SIMD broadcast.  Allows flexible sizing of TILE width (TILE_N) 
// by dynamically selecting one of two code paths: one uses TILE_N = 32 and 
// the other uses TILE_N = 8, 16, or 24.
#define TILE_M          2
#define TILE_K          KERNEL_WIDTH
#define TILE_N          32

#ifndef __BEIGNET__
__attribute__((intel_reqd_sub_group_size(8)))
#endif
__kernel void Conv_Interleaved(GEMM_LIKE_KERNEL_ARGS)
{
    const int group_x = get_group_id(0);
    const int group_y = get_group_id(1);
    const int global_x = get_global_id(0);
    const int global_y = get_global_id(1);
    const int global_z = get_global_id(2);
    int interleaved_y;
    int kernel_y;
    int kernel_idx;
    
#define DOT_PRODUCT_8( _result, _rowA, colB )    \
    {   \
        _result.s0 = mad( _rowA, sub_group_broadcast( colB, 0 ), _result.s0 );  \
        _result.s1 = mad( _rowA, sub_group_broadcast( colB, 1 ), _result.s1 );  \
        _result.s2 = mad( _rowA, sub_group_broadcast( colB, 2 ), _result.s2 );  \
        _result.s3 = mad( _rowA, sub_group_broadcast( colB, 3 ), _result.s3 );  \
        _result.s4 = mad( _rowA, sub_group_broadcast( colB, 4 ), _result.s4 );  \
        _result.s5 = mad( _rowA, sub_group_broadcast( colB, 5 ), _result.s5 );  \
        _result.s6 = mad( _rowA, sub_group_broadcast( colB, 6 ), _result.s6 );  \
        _result.s7 = mad( _rowA, sub_group_broadcast( colB, 7 ), _result.s7 );  \
    }
        typedef CAT( Dtype, KERNEL_WIDTH ) Dtype_t;

    // True for all threads if filter_width is multiple of TILE_N
    // else, true for all but right-most column of threads.
    if( TILE_N_LAST == 0 || global_x < WIDTH1 / TILE_N ) 
    {
        // Result ctile (*dst) is M rows x N columns
        // LWG size is 1x8.  Thus each thread calculates 8*M rows x N cols of ctile.
        Dtype8  blockC00 = 0.f;
        Dtype8  blockC10 = 0.f;
        Dtype8  blockC20 = 0.f;
        Dtype8  blockC30 = 0.f;
        Dtype8  blockC01 = 0.f;
        Dtype8  blockC11 = 0.f;
        Dtype8  blockC21 = 0.f;
        Dtype8  blockC31 = 0.f;

        // Src0 (patch input) is directly used as atile.
        // Each work item points to the start of a different patch.
        // atile is M rows x K columns.
        int curr_x0 = ( ( global_y * TILE_M + 0 ) % output_width ) * STRIDE_X;
        int curr_x1 = ( ( global_y * TILE_M + 1 ) % output_width ) * STRIDE_X;
        int curr_y0 = ( ( global_y * TILE_M + 0 ) / output_width ) * STRIDE_Y;
        int curr_y1 = ( ( global_y * TILE_M + 1 ) / output_width ) * STRIDE_Y;
#if INPUT_PAD_H != 0 || INPUT_PAD_W != 0 || DILATION_X != 1 || DILATION_Y != 1
        int saved_y0 = curr_y0;
        int saved_y1 = curr_y1;
#endif
        const __global Dtype *src0_read0 = src0
         + aligned_input_size * global_z                                            // batch offset
         + (curr_y0 - INPUT_PAD_H) * ROW_PITCH   // y offset
         + curr_x0 - INPUT_PAD_W;                // x offset
        const __global Dtype *src0_read1 = src0
         + aligned_input_size * global_z                                            // batch offset
         + (curr_y1 - INPUT_PAD_H) * ROW_PITCH   // y offset
         + curr_x1 - INPUT_PAD_W;                // x offset

        // Src1 (filter) is directly used as btile.
        // It starts at the top of src1 and walks down.
        // btile is K rows x N columns.
        const __global Dtype *src1_read = src1 + ( global_x * TILE_N * 2);

        // Walk DOWN src0 (patch 0, 1, 2, ...) and DOWN src1.
        // Inner loop loads and FMADs one row (KERNEL_WIDTH) of each input patch 
        // and KERNEL_WIDTH/2 rows of interleaved filter.
        int patch_depth = 0;
        do
        {
            int patch_row = 0;
            do
            {
                // Load atile and btile.
                // Kernel data is partially interleaved.  Every 2 rows are interleaved at Dtype8 granularity.
                // The exception is that if KERNEL_WIDTH is odd the last row is not interleaved.  The non 
                // interleaved row is padded with zero to ensure same size as interleaved rows. This
                // interleaving is done to ensure 0% GDR bank conflicts.  For example, this is how the
                // kernel data would be arranged before/after interleaving for KERNEL_WIDTH=3.
                // (0, 0) (8, 0) (16, 0) (24, 0) ...       (0, 0) (0, 1) (8, 0) (0, 1) (16, 0) (0, 1) (24, 0) ..
                // (0, 1) (8, 1) (16, 1) (24, 1) ... =>    (0, 2) (8, 2) (16, 2) (24, 2) ...
                // (0, 2) (8, 2) (16, 2) (24, 2) ...       ...
                // ...
                const bool kernel_width_is_odd = KERNEL_WIDTH % 2 == 1;
#if INPUT_PAD_H == 0 && INPUT_PAD_W == 0 && DILATION_X == 1 && DILATION_Y == 1
                Dtype_t blockA00 = ( (const __global Dtype_t*)src0_read0 )[  0  ]; src0_read0 += ROW_PITCH;
                Dtype_t blockA01 = ( (const __global Dtype_t*)src0_read1 )[  0  ]; src0_read1 += ROW_PITCH;
                Dtype*  pblockA00 = (Dtype*)(&blockA00);
                Dtype*  pblockA01 = (Dtype*)(&blockA01);
#else
                Dtype_t blockA00;
                Dtype*  pblockA00 = (Dtype*)(&blockA00);
                int pos = 0;
                LOOP(KERNEL_WIDTH, pos,
                {
                  if (curr_y0 >= INPUT_PAD_H && curr_y0 < input_height + INPUT_PAD_H && curr_x0 + pos * DILATION_X >= INPUT_PAD_W && curr_x0 + pos * DILATION_X < input_width + INPUT_PAD_W)
                    pblockA00[pos] = src0_read0[pos * DILATION_X];
                  else
                    pblockA00[pos] = 0;
                })
                curr_y0 += DILATION_Y;
                Dtype_t blockA01;
                Dtype*  pblockA01 = (Dtype*)(&blockA01);
                pos = 0;
                LOOP(KERNEL_WIDTH, pos,
                {
                  if (curr_y1 >= INPUT_PAD_H && curr_y1 < input_height + INPUT_PAD_H && curr_x1 + pos * DILATION_X >= INPUT_PAD_W && curr_x1 + pos * DILATION_X < input_width + INPUT_PAD_W)
                    pblockA01[pos] = src0_read1[pos * DILATION_X];
                  else
                    pblockA01[pos] = 0;
                })
                curr_y1 += DILATION_Y;
                src0_read0 += (ROW_PITCH * DILATION_Y);
                src0_read1 += (ROW_PITCH * DILATION_Y);
#endif
                Dtype blockB00[KERNEL_WIDTH*4];
                Dtype8* p8BlockB00 = (Dtype8*)blockB00;
                Dtype4* p4BlockB00 = (Dtype4*)blockB00;
                Dtype*  pBlockB00 =  (Dtype* )blockB00;

                interleaved_y = 0;
                LOOP(KERNEL_WIDTH_DIV2, interleaved_y, 
                { 
                    p8BlockB00[interleaved_y] = as_Dtype8( SUB_GROUP_BLOCK_READ8( (const __global INT_TYPE*)src1_read ) );
                    src1_read += WIDTH1 * 2;
                } )
                if ( kernel_width_is_odd )
                {
                    p4BlockB00[KERNEL_WIDTH - 1] = as_Dtype4( SUB_GROUP_BLOCK_READ4( (const __global INT_TYPE*)src1_read ) );
                    src1_read += WIDTH1 * 2;
                }
                // Perform MADs
                kernel_idx = 0;
                interleaved_y = 0;
                LOOP(KERNEL_WIDTH_DIV2, interleaved_y, 
                {
                    kernel_y = interleaved_y * 2;
                    DOT_PRODUCT_8( blockC00, pblockA00[kernel_y    ], pBlockB00[kernel_idx] );
                    DOT_PRODUCT_8( blockC01, pblockA01[kernel_y    ], pBlockB00[kernel_idx] ); kernel_idx++;
                    DOT_PRODUCT_8( blockC00, pblockA00[kernel_y + 1], pBlockB00[kernel_idx] );
                    DOT_PRODUCT_8( blockC01, pblockA01[kernel_y + 1], pBlockB00[kernel_idx] ); kernel_idx++;
                    DOT_PRODUCT_8( blockC10, pblockA00[kernel_y    ], pBlockB00[kernel_idx] );
                    DOT_PRODUCT_8( blockC11, pblockA01[kernel_y    ], pBlockB00[kernel_idx] ); kernel_idx++;
                    DOT_PRODUCT_8( blockC10, pblockA00[kernel_y + 1], pBlockB00[kernel_idx] );
                    DOT_PRODUCT_8( blockC11, pblockA01[kernel_y + 1], pBlockB00[kernel_idx] ); kernel_idx++;
                    DOT_PRODUCT_8( blockC20, pblockA00[kernel_y    ], pBlockB00[kernel_idx] );
                    DOT_PRODUCT_8( blockC21, pblockA01[kernel_y    ], pBlockB00[kernel_idx] ); kernel_idx++;
                    DOT_PRODUCT_8( blockC20, pblockA00[kernel_y + 1], pBlockB00[kernel_idx] );
                    DOT_PRODUCT_8( blockC21, pblockA01[kernel_y + 1], pBlockB00[kernel_idx] ); kernel_idx++;
                    DOT_PRODUCT_8( blockC30, pblockA00[kernel_y    ], pBlockB00[kernel_idx] );
                    DOT_PRODUCT_8( blockC31, pblockA01[kernel_y    ], pBlockB00[kernel_idx] ); kernel_idx++;
                    DOT_PRODUCT_8( blockC30, pblockA00[kernel_y + 1], pBlockB00[kernel_idx] );
                    DOT_PRODUCT_8( blockC31, pblockA01[kernel_y + 1], pBlockB00[kernel_idx] ); kernel_idx++;
                } )
                if ( kernel_width_is_odd )
                {
                    kernel_y = interleaved_y * 2;
                    DOT_PRODUCT_8( blockC00, pblockA00[kernel_y], pBlockB00[kernel_idx] );
                    DOT_PRODUCT_8( blockC01, pblockA01[kernel_y], pBlockB00[kernel_idx] ); kernel_idx++;
                    DOT_PRODUCT_8( blockC10, pblockA00[kernel_y], pBlockB00[kernel_idx] );
                    DOT_PRODUCT_8( blockC11, pblockA01[kernel_y], pBlockB00[kernel_idx] ); kernel_idx++;
                    DOT_PRODUCT_8( blockC20, pblockA00[kernel_y], pBlockB00[kernel_idx] );
                    DOT_PRODUCT_8( blockC21, pblockA01[kernel_y], pBlockB00[kernel_idx] ); kernel_idx++;
                    DOT_PRODUCT_8( blockC30, pblockA00[kernel_y], pBlockB00[kernel_idx] );
                    DOT_PRODUCT_8( blockC31, pblockA01[kernel_y], pBlockB00[kernel_idx] ); kernel_idx++;
                }
            }

            //while( ++patch_row < 1 ); //debug
            while( ++patch_row < KERNEL_HEIGHT );
#if INPUT_PAD_W != 0 || INPUT_PAD_H != 0 || DILATION_X != 1 || DILATION_Y != 1
            curr_y0 = saved_y0;
            curr_y1 = saved_y1;
#endif
            src0_read0 += slice_pitch - ( KERNEL_HEIGHT * ROW_PITCH * DILATION_Y ); // reset to start of next slice of patch
            src0_read1 += slice_pitch - ( KERNEL_HEIGHT * ROW_PITCH * DILATION_Y );
        } 
        //while ( ++patch_depth < 1 );  //debug
        while ( ++patch_depth < INPUT_DEPTH );

        // Dst resembles a cube of width x height x (output channel * batches).  Each tile writes: 
        // (SIMD * TILE_M) x 1 x TILE_N.  Partial writes most likely generated if padding used.
        int_tp out0_offset = global_z * out_pitch_z                                                       // batch offset
         + ( group_x * TILE_N ) * out_pitch_y                                           // channel offset
         + ( ( global_y * TILE_M + 0 ) / output_width + OUT_PADDING_HEIGHT ) * OUT_PITCH_X // y offset
         + ( ( global_y * TILE_M + 0 ) % output_width ) + OUT_PADDING_LEFT;               // x offset
        int_tp out1_offset = global_z * out_pitch_z                                                       // batch offset
         + ( group_x * TILE_N ) * out_pitch_y                                           // channel offset
         + ( ( global_y * TILE_M + 1 ) / output_width + OUT_PADDING_HEIGHT ) * OUT_PITCH_X // y offset
         + ( ( global_y * TILE_M + 1 ) % output_width ) + OUT_PADDING_LEFT;               // x offset

        Dtype bias[4];
        Dtype4 *bias_vec;
        bias_vec = (Dtype4*)bias;
        *bias_vec = as_Dtype4(SUB_GROUP_BLOCK_READ4((__global INT_TYPE *)biases + group_x * TILE_N));

        if( global_y * TILE_M < output_width * output_height )
        {
            for( int i = 0; i < 8; i++ )
            {
                ACTIVATION_FUNCTION(dst, out0_offset + ( 0+i) * out_pitch_y, blockC00[i] + intel_sub_group_shuffle(bias[0], i));
                ACTIVATION_FUNCTION(dst, out0_offset + ( 8+i) * out_pitch_y, blockC10[i] + intel_sub_group_shuffle(bias[1], i));
                ACTIVATION_FUNCTION(dst, out0_offset + (16+i) * out_pitch_y, blockC20[i] + intel_sub_group_shuffle(bias[2], i));
                ACTIVATION_FUNCTION(dst, out0_offset + (24+i) * out_pitch_y, blockC30[i] + intel_sub_group_shuffle(bias[3], i));
            }
        }
        if( global_y * TILE_M + 1 < output_width * output_height )
        {
            for( int i = 0; i < 8; i++ )
            {
                ACTIVATION_FUNCTION(dst, out1_offset + ( 0+i) * out_pitch_y, blockC01[i] + intel_sub_group_shuffle(bias[0], i));
                ACTIVATION_FUNCTION(dst, out1_offset + ( 8+i) * out_pitch_y, blockC11[i] + intel_sub_group_shuffle(bias[1], i));
                ACTIVATION_FUNCTION(dst, out1_offset + (16+i) * out_pitch_y, blockC21[i] + intel_sub_group_shuffle(bias[2], i));
                ACTIVATION_FUNCTION(dst, out1_offset + (24+i) * out_pitch_y, blockC31[i] + intel_sub_group_shuffle(bias[3], i));
            }
        }
    }    
#if TILE_N_LAST > 0
    else
    {
        
        // Result ctile (*dst) is M rows x N columns
        // LWG size is 1x8.  Thus each thread calculates 8*M rows x N cols of ctile.
        int i = 0;
        Dtype8  blockC0[TILE_N_LAST_DIV8];
        Dtype8  blockC1[TILE_N_LAST_DIV8];
        LOOP(TILE_N_LAST_DIV8, i,
        {
            blockC0[i] = 0.f;
            blockC1[i] = 0.f;
        } )
        
        // Src0 (patch input) is directly used as atile.
        // Each work item points to the start of a different patch.
        // atile is M rows x K columns.
        int curr_x0 = ( ( global_y * TILE_M + 0 ) % output_width ) * STRIDE_X;
        int curr_x1 = ( ( global_y * TILE_M + 1 ) % output_width ) * STRIDE_X;
        int curr_y0 = ( ( global_y * TILE_M + 0 ) / output_width ) * STRIDE_Y;
        int curr_y1 = ( ( global_y * TILE_M + 1 ) / output_width ) * STRIDE_Y;
#if INPUT_PAD_H != 0 || INPUT_PAD_W != 0 || DILATION_X != 1 || DILATION_Y != 1
        int saved_y0 = curr_y0;
        int saved_y1 = curr_y1;
#endif
        const __global Dtype *src0_read0 = src0
         + aligned_input_size * global_z                                            // batch offset
         + (curr_y0 - INPUT_PAD_H) * ROW_PITCH   // y offset
         + curr_x0 - INPUT_PAD_W;                // x offset
        const __global Dtype *src0_read1 = src0
         + aligned_input_size * global_z                                            // batch offset
         + (curr_y1 - INPUT_PAD_H) * ROW_PITCH   // y offset
         + curr_x1 - INPUT_PAD_W;                // x offset

        // Src1 (filter) is directly used as btile.
        // It starts at the top of src1 and walks down.
        // btile is K rows x N columns.
        const __global Dtype *src1_read = src1 + ( global_x * TILE_N  * 2);

        // Walk DOWN src0 (patch 0, 1, 2, ...) and DOWN src1.
        // Inner loop loads and FMADs one row (KERNEL_WIDTH) of each input patch 
        // and KERNEL_WIDTH/2 rows of interleaved filter.
        int patch_depth = 0;
        do
        {
            int patch_row = 0;
            do
            {
                // Load atile and interleaved btile.
                const bool kernel_width_is_odd = KERNEL_WIDTH % 2 == 1;
#if INPUT_PAD_H == 0 && INPUT_PAD_W == 0 && DILATION_X == 1 && DILATION_Y == 1
                Dtype_t blockA00 = ( (const __global Dtype_t*)src0_read0 )[  0  ]; src0_read0 += ROW_PITCH;
                Dtype_t blockA01 = ( (const __global Dtype_t*)src0_read1 )[  0  ]; src0_read1 += ROW_PITCH;
                Dtype*  pblockA00 = (Dtype*)(&blockA00);
                Dtype*  pblockA01 = (Dtype*)(&blockA01);
#else
                Dtype_t blockA00;
                Dtype*  pblockA00 = (Dtype*)(&blockA00);
                int pos = 0;
                LOOP(KERNEL_WIDTH, pos,
                {
                  if (curr_y0 >= INPUT_PAD_H && curr_y0 < input_height + INPUT_PAD_H && curr_x0 + pos * DILATION_X >= INPUT_PAD_W && curr_x0 + pos * DILATION_X < input_width + INPUT_PAD_W)
                    pblockA00[pos] = src0_read0[pos * DILATION_X];
                  else
                    pblockA00[pos] = 0;
                })
                curr_y0 += DILATION_Y;
                Dtype_t blockA01;
                Dtype*  pblockA01 = (Dtype*)(&blockA01);
                pos = 0;
                LOOP(KERNEL_WIDTH, pos,
                {
                  if (curr_y1 >= INPUT_PAD_H && curr_y1 < input_height + INPUT_PAD_H && curr_x1 + pos * DILATION_X >= INPUT_PAD_W && curr_x1 + pos * DILATION_X < input_width + INPUT_PAD_W)
                    pblockA01[pos] = src0_read1[pos * DILATION_X];
                  else
                    pblockA01[pos] = 0;
                })
                curr_y1 += DILATION_Y;
                src0_read0 += (ROW_PITCH * DILATION_Y);
                src0_read1 += (ROW_PITCH * DILATION_Y);
#endif
                Dtype blockB[KERNEL_WIDTH * TILE_N_LAST_DIV8];

                interleaved_y = 0;
                LOOP(KERNEL_WIDTH_DIV2, interleaved_y, 
                { 
#if TILE_N_LAST_DIV8 == 1
                    Dtype2* p2BlockB = (Dtype2* )blockB;
                    p2BlockB[interleaved_y] = as_Dtype2( SUB_GROUP_BLOCK_READ2( (const __global INT_TYPE*)src1_read ) );
#elif TILE_N_LAST_DIV8 == 2
                    Dtype4* p4BlockB = (Dtype4* )blockB;
                    p4BlockB[interleaved_y] = as_Dtype4( SUB_GROUP_BLOCK_READ4( (const __global INT_TYPE*)src1_read ) );
#elif TILE_N_LAST_DIV8 == 3
                    //TODO: broken.  No block_read6
                    Dtype6* p6BlockB = (Dtype6* )blockB;
                    (*((Dtype8*)(&p6BlockB[interleaved_y]))).s0123 = as_Dtype4( SUB_GROUP_BLOCK_READ4( (const __global INT_TYPE*)src1_read ) );
                    (*((Dtype8*)(&p6BlockB[interleaved_y]))).s45 = as_Dtype2( SUB_GROUP_BLOCK_READ2( (const __global INT_TYPE*)(src1_read + 4 * 8) ) );
#endif
                    src1_read += WIDTH1 * 2;
                } )
                if ( kernel_width_is_odd )
                {
#if TILE_N_LAST_DIV8 == 1
                    Dtype* pBlockB = (Dtype* )blockB;
                    pBlockB[KERNEL_WIDTH - 1] = as_Dtype( SUB_GROUP_BLOCK_READ( (const __global INT_TYPE*)src1_read ) );
#elif TILE_N_LAST_DIV8 == 2
                    Dtype2* p2BlockB = (Dtype2* )blockB;
                    p2BlockB[KERNEL_WIDTH - 1] = as_Dtype2( SUB_GROUP_BLOCK_READ2( (const __global INT_TYPE*)src1_read ) );
#elif TILE_N_LAST_DIV8 == 3
                    Dtype3* p3BlockB = (Dtype3* )blockB;
                    p3BlockB[KERNEL_WIDTH - 1].s01 = as_Dtype2( SUB_GROUP_BLOCK_READ2( (const __global INT_TYPE*)src1_read ) );
                    p3BlockB[KERNEL_WIDTH - 1].s2 = as_Dtype( SUB_GROUP_BLOCK_READ( (const __global INT_TYPE*) (src1_read + 8) ) );
#endif
                    src1_read += WIDTH1 * 2;
                }

                // Perform MADs
                Dtype* pBlockB = (Dtype*)blockB;
                kernel_idx = 0;
                interleaved_y = 0;
                LOOP(KERNEL_WIDTH_DIV2, interleaved_y, 
                {
                    kernel_y = interleaved_y * 2;
                    DOT_PRODUCT_8( blockC0[0], pblockA00[kernel_y    ], pBlockB[kernel_idx] );
                    DOT_PRODUCT_8( blockC1[0], pblockA01[kernel_y    ], pBlockB[kernel_idx] ); kernel_idx++;
                    DOT_PRODUCT_8( blockC0[0], pblockA00[kernel_y + 1], pBlockB[kernel_idx] );
                    DOT_PRODUCT_8( blockC1[0], pblockA01[kernel_y + 1], pBlockB[kernel_idx] ); kernel_idx++;
#if TILE_N_LAST_DIV8 >= 2
                    DOT_PRODUCT_8( blockC0[1], pblockA00[kernel_y    ], pBlockB[kernel_idx] );
                    DOT_PRODUCT_8( blockC1[1], pblockA01[kernel_y    ], pBlockB[kernel_idx] ); kernel_idx++;
                    DOT_PRODUCT_8( blockC0[1], pblockA00[kernel_y + 1], pBlockB[kernel_idx] );
                    DOT_PRODUCT_8( blockC1[1], pblockA01[kernel_y + 1], pBlockB[kernel_idx] ); kernel_idx++;
#if TILE_N_LAST_DIV8 >= 3
                    DOT_PRODUCT_8( blockC0[2], pblockA00[kernel_y    ], pBlockB[kernel_idx] );
                    DOT_PRODUCT_8( blockC1[2], pblockA01[kernel_y    ], pBlockB[kernel_idx] ); kernel_idx++;
                    DOT_PRODUCT_8( blockC0[2], pblockA00[kernel_y + 1], pBlockB[kernel_idx] );
                    DOT_PRODUCT_8( blockC1[2], pblockA01[kernel_y + 1], pBlockB[kernel_idx] ); kernel_idx++;
#endif
#endif
                } )
                    kernel_y = interleaved_y * 2;
                if ( kernel_width_is_odd )
                {
                    DOT_PRODUCT_8( blockC0[0], pblockA00[kernel_y], pBlockB[kernel_idx] );
                    DOT_PRODUCT_8( blockC1[0], pblockA01[kernel_y], pBlockB[kernel_idx] ); kernel_idx++;
#if TILE_N_LAST_DIV8 >= 2
                    DOT_PRODUCT_8( blockC0[1], pblockA00[kernel_y], pBlockB[kernel_idx] );
                    DOT_PRODUCT_8( blockC1[1], pblockA01[kernel_y], pBlockB[kernel_idx] ); kernel_idx++;
#if TILE_N_LAST_DIV8 >= 3
                    DOT_PRODUCT_8( blockC0[2], pblockA00[kernel_y], pBlockB[kernel_idx] );
                    DOT_PRODUCT_8( blockC1[2], pblockA01[kernel_y], pBlockB[kernel_idx] ); kernel_idx++;
#endif
#endif
                }
            }

            //while( ++patch_row < 1 ); //debug
            while( ++patch_row < KERNEL_HEIGHT );
#if INPUT_PAD_W != 0 || INPUT_PAD_H != 0 || DILATION_X != 1 || DILATION_Y != 1
            curr_y0 = saved_y0;
            curr_y1 = saved_y1;
#endif
            src0_read0 += slice_pitch - ( KERNEL_HEIGHT * ROW_PITCH * DILATION_Y ); // reset to start of next slice of patch
            src0_read1 += slice_pitch - ( KERNEL_HEIGHT * ROW_PITCH * DILATION_Y );
        } 
        //while ( ++patch_depth < 1 );  //debug
        while ( ++patch_depth < INPUT_DEPTH );

        // Dst resembles a cube of width x height x (output channel * batches).  Each tile writes: 
        // (SIMD * TILE_M) x 1 x TILE_N.  Partial writes most likely generated if padding used.
        int_tp out0_offset = global_z * out_pitch_z                                                       // batch offset
         + ( group_x * TILE_N ) * out_pitch_y                                           // channel offset
         + ( ( global_y * TILE_M + 0 ) / output_width + OUT_PADDING_HEIGHT ) * OUT_PITCH_X // y offset
         + ( ( global_y * TILE_M + 0 ) % output_width ) + OUT_PADDING_LEFT;               // x offset
        int_tp out1_offset = global_z * out_pitch_z                                                       // batch offset
         + ( group_x * TILE_N ) * out_pitch_y                                           // channel offset
         + ( ( global_y * TILE_M + 1 ) / output_width + OUT_PADDING_HEIGHT ) * OUT_PITCH_X // y offset
         + ( ( global_y * TILE_M + 1 ) % output_width ) + OUT_PADDING_LEFT;               // x offset
        __global Dtype *out1 = dst + out1_offset;

        Dtype bias[4];
        Dtype4 *bias_vec;
        bias_vec = (Dtype4*)bias;
        *bias_vec = as_Dtype4(SUB_GROUP_BLOCK_READ4((__global INT_TYPE *)biases + group_x * TILE_N));
        if( global_y * TILE_M < output_width * output_height )
        {
            for( int i = 0; i < 8; i++ )
            {
                if ( TILE_N_LAST_DIV8 > 0 ) ACTIVATION_FUNCTION(dst, out0_offset + ( 0+i) * out_pitch_y, blockC0[0][i] + intel_sub_group_shuffle(bias[0], i));
                if ( TILE_N_LAST_DIV8 > 1 ) ACTIVATION_FUNCTION(dst, out0_offset + ( 8+i) * out_pitch_y, blockC0[1][i] + intel_sub_group_shuffle(bias[1], i));
                if ( TILE_N_LAST_DIV8 > 2 ) ACTIVATION_FUNCTION(dst, out0_offset + (16+i) * out_pitch_y, blockC0[2][i] + intel_sub_group_shuffle(bias[2], i));
                if ( TILE_N_LAST_DIV8 > 3 ) ACTIVATION_FUNCTION(dst, out0_offset + (24+i) * out_pitch_y, blockC0[3][i] + intel_sub_group_shuffle(bias[3], i));
            }
        }
        if( global_y * TILE_M + 1 < output_width * output_height )
        {
            for( int i = 0; i < 8; i++ )
            {
                if ( TILE_N_LAST_DIV8 > 0 ) ACTIVATION_FUNCTION(dst, out1_offset + ( 0+i) * out_pitch_y, blockC1[0][i] + intel_sub_group_shuffle(bias[0], i));
                if ( TILE_N_LAST_DIV8 > 1 ) ACTIVATION_FUNCTION(dst, out1_offset + ( 8+i) * out_pitch_y, blockC1[1][i] + intel_sub_group_shuffle(bias[1], i));
                if ( TILE_N_LAST_DIV8 > 2 ) ACTIVATION_FUNCTION(dst, out1_offset + (16+i) * out_pitch_y, blockC1[2][i] + intel_sub_group_shuffle(bias[2], i));
                if ( TILE_N_LAST_DIV8 > 3 ) ACTIVATION_FUNCTION(dst, out1_offset + (24+i) * out_pitch_y, blockC1[3][i] + intel_sub_group_shuffle(bias[3], i));
            }
        }
    }
#endif
}
#endif
#if defined(GEMM_LIKE_CONV_32_2_SIMD16) || defined(GEMM_LIKE_CONV_32_1_SIMD16)

#define INTERLEAVED_SIMD16_OUTPUT(_out_, _offset_,  _m_) do {\
    if (global_y * TILE_M < output_width * output_height ) \
    { \
      if ( ( OUT_DEPTH % TILE_N ) == 0 ) {\
        for (int i = 0; i < 16; i++) \
        { \
          ACTIVATION_FUNCTION(_out_, _offset_ + ( 0+i) * out_pitch_y, blockC0 ##_m_ [i] + intel_sub_group_shuffle(bias[0], i)); \
          ACTIVATION_FUNCTION(_out_, _offset_ + (16+i) * out_pitch_y, blockC1 ##_m_ [i] + intel_sub_group_shuffle(bias[1], i)); \
        } \
      } \
      else if( ( OUT_DEPTH % 16 ) == 0 ) { \
        if ( ( global_x + 1 ) < get_global_size(0) ) { \
          for ( int i = 0; i < 16; i++ ) \
          { \
            ACTIVATION_FUNCTION(_out_, _offset_ + ( 0+i) * out_pitch_y, blockC0 ##_m_ [i] + intel_sub_group_shuffle(bias[0], i)); \
            ACTIVATION_FUNCTION(_out_, _offset_ + (16+i) * out_pitch_y, blockC1 ##_m_ [i] + intel_sub_group_shuffle(bias[1], i)); \
          } \
        } \
        else { \
          for (int i = 0; i < 16; i++) \
          { \
            ACTIVATION_FUNCTION(_out_, _offset_ + ( 0+i) * out_pitch_y, blockC0 ##_m_ [i] + intel_sub_group_shuffle(bias[0], i)); \
          } \
        } \
      } \
      else { \
        if ( ( global_x + 1 ) < get_global_size(0) ) \
        { \
          for ( int i = 0; i < 16; i++ ) \
          { \
            ACTIVATION_FUNCTION(_out_, _offset_ + ( 0+i) * out_pitch_y, blockC0 ##_m_[i] + intel_sub_group_shuffle(bias[0], i)); \
            ACTIVATION_FUNCTION(_out_, _offset_ + (16+i) * out_pitch_y, blockC1 ##_m_[i] + intel_sub_group_shuffle(bias[1], i)); \
          } \
        } \
        else { \
          if ( (OUT_DEPTH % TILE_N) > 16 ) { \
            for (int i = 0; i < 16 ; i++) \
            { \
              ACTIVATION_FUNCTION(_out_, _offset_ + ( 0+i) * out_pitch_y, blockC0 ##_m_[i] + intel_sub_group_shuffle(bias[0], i)); \
            } \
            for (int i = 0; i < OUT_DEPTH % 16 ; i++) \
            { \
              ACTIVATION_FUNCTION(_out_, _offset_ + (16+i) * out_pitch_y, blockC1 ##_m_[i] + intel_sub_group_shuffle(bias[1], i)); \
            } \
          } \
          else { \
            for (int i = 0; i < OUT_DEPTH % 16 ; i++) \
            { \
              ACTIVATION_FUNCTION(_out_, _offset_ + ( 0+i) * out_pitch_y, blockC0 ##_m_[i] + intel_sub_group_shuffle(bias[0], i)); \
            } \
          } \
        } \
      } \
    } \
 }while(0)
#endif

#ifdef GEMM_LIKE_CONV_32_1_SIMD16
#define TILE_M          1
#define TILE_K          KERNEL_WIDTH
#define TILE_N          32

#ifndef __BEIGNET__
__attribute__((intel_reqd_sub_group_size(16)))
#endif
__kernel void Conv_Interleaved(GEMM_LIKE_KERNEL_ARGS)
{
    const int group_x = get_group_id(0);
    const int group_y = get_group_id(1);
    const int global_x = get_global_id(0);
    const int global_y = get_global_id(1);
    const int global_z = get_global_id(2);
    int interleaved_y;
    int kernel_y;
    int kernel_idx;

    // Result ctile (*dst) is M rows x N columns
    // LWG size is 1x16.  Thus each thread calculates 16*M rows x N cols of ctile.
    Dtype16  blockC00 = 0.f;
    Dtype16  blockC10 = 0.f;

    // Src0 (patch input) is directly used as atile.
    // Each work item points to the start of a different patch.
    // atile is M rows x K columns.
    int curr_x = ( global_y % output_width ) * STRIDE_X;
    int curr_y = ( global_y / output_width ) * STRIDE_Y;
#if INPUT_PAD_H != 0 || INPUT_PAD_W != 0 || DILATION_X != 1 || DILATION_Y != 1
    int saved_y = curr_y;
#endif
    const __global Dtype *src0_read = src0
     + aligned_input_size * global_z                            // batch offset
     + (curr_y - INPUT_PAD_H) * ROW_PITCH      // y offset
     + curr_x - INPUT_PAD_W;                 // x offset
     const __global Dtype *src0_read_orig = src0_read;

    // Src1 (filter) is directly used as btile.
    // It starts at the top of src1 and walks down.
    // btile is K rows x N columns.
    const __global Dtype *src1_read = src1 + ( global_x * TILE_N * 2 );

#define DOT_PRODUCT_16( _result, _rowA, colB )    \
    {   \
        _result.s0 = mad( _rowA, sub_group_broadcast( colB,  0 ), _result.s0 );  \
        _result.s1 = mad( _rowA, sub_group_broadcast( colB,  1 ), _result.s1 );  \
        _result.s2 = mad( _rowA, sub_group_broadcast( colB,  2 ), _result.s2 );  \
        _result.s3 = mad( _rowA, sub_group_broadcast( colB,  3 ), _result.s3 );  \
        _result.s4 = mad( _rowA, sub_group_broadcast( colB,  4 ), _result.s4 );  \
        _result.s5 = mad( _rowA, sub_group_broadcast( colB,  5 ), _result.s5 );  \
        _result.s6 = mad( _rowA, sub_group_broadcast( colB,  6 ), _result.s6 );  \
        _result.s7 = mad( _rowA, sub_group_broadcast( colB,  7 ), _result.s7 );  \
        _result.s8 = mad( _rowA, sub_group_broadcast( colB,  8 ), _result.s8 );  \
        _result.s9 = mad( _rowA, sub_group_broadcast( colB,  9 ), _result.s9 );  \
        _result.sa = mad( _rowA, sub_group_broadcast( colB, 10 ), _result.sa );  \
        _result.sb = mad( _rowA, sub_group_broadcast( colB, 11 ), _result.sb );  \
        _result.sc = mad( _rowA, sub_group_broadcast( colB, 12 ), _result.sc );  \
        _result.sd = mad( _rowA, sub_group_broadcast( colB, 13 ), _result.sd );  \
        _result.se = mad( _rowA, sub_group_broadcast( colB, 14 ), _result.se );  \
        _result.sf = mad( _rowA, sub_group_broadcast( colB, 15 ), _result.sf );  \
    }
    typedef CAT( Dtype, KERNEL_WIDTH ) Dtype_t;
    // Walk DOWN src0 (patch 0, 1, 2, ...) and DOWN src1.
    // Inner loop loads and FMADs one row (KERNEL_WIDTH) of each input patch
    // and KERNEL_WIDTH/2 rows of interleaved filter.
    int patch_depth = 0;
#ifndef __BEIGNET__
    __attribute__((opencl_unroll_hint(1)))
#endif
    do
    {
        int patch_row = 0;
#if INPUT_PAD_H != 0 || INPUT_PAD_W != 0 || DILATION_X != 1 || DILATION_Y != 1
        curr_y = saved_y;
#endif
#ifndef __BEIGNET__
        __attribute__((opencl_unroll_hint(1)))
#endif
        do
        {
            // Load atile and btile.
            // Kernel data is partially interleaved.  Every 2 rows are interleaved at Dtype16 granularity.
            // The exception is that if KERNEL_WIDTH is odd the last row is not interleaved.  The non
            // interleaved row is padded with zero to ensure same size as interleaved rows. This
            // interleaving is done to ensure 0% GDR bank conflicts.  For example, this is how the
            // kernel data would be arranged before/after interleaving for KERNEL_WIDTH=3.
            // (0, 0) (16, 0) (32, 0) (48, 0) ...     (0, 0) ( 0, 1) (16, 0) ( 0, 1) (32, 0) (0, 1) (48, 0) ...
            // (0, 1) (16, 1) (32, 1) (48, 1) ... =>  (0, 2) (16, 2) (32, 2) (48, 2) ...
            // (0, 2) (16, 2) (32, 2) (48, 2) ...     ...
            // ...
            const bool kernel_width_is_odd = KERNEL_WIDTH % 2 == 1;

#if INPUT_PAD_W == 0 && INPUT_PAD_H == 0 && DILATION_X == 1 && DILATION_Y == 1
            Dtype_t blockA00 = ( (const __global Dtype_t*)src0_read )[  0  ];
            Dtype*  pblockA00 = (Dtype*)(&blockA00);
#else
            Dtype_t blockA00;
            Dtype*  pblockA00 = (Dtype*)(&blockA00);
            int pos = 0;
            LOOP(KERNEL_WIDTH, pos,
            {
              if (curr_y >= INPUT_PAD_H && curr_y < input_height + INPUT_PAD_H && curr_x + pos * DILATION_X >= INPUT_PAD_W && curr_x + pos * DILATION_X < input_width + INPUT_PAD_W)
                pblockA00[pos] = src0_read[pos * DILATION_X];
              else
                pblockA00[pos] = 0;
            })
            curr_y += DILATION_Y;
#endif
            src0_read += ROW_PITCH * DILATION_Y;
            INT_TYPE blockB00[KERNEL_WIDTH * 2];
            INT_TYPE4* p4BlockB00 = (INT_TYPE4*)blockB00;
            INT_TYPE2* p2BlockB00 = (INT_TYPE2*)blockB00;
            Dtype* pBlockB00  = (Dtype*)blockB00;
            interleaved_y = 0;
            LOOP(KERNEL_WIDTH_DIV2, interleaved_y,
            {
                p4BlockB00[interleaved_y] = SUB_GROUP_BLOCK_READ4( (const __global INT_TYPE*)src1_read );
                src1_read += WIDTH1 * 2;
            } )
            if ( kernel_width_is_odd )
            {
                p2BlockB00[KERNEL_WIDTH - 1] = SUB_GROUP_BLOCK_READ2( (const __global INT_TYPE*)src1_read );
                src1_read += WIDTH1 * 2;
            }

            // Perform MADs
            kernel_idx = 0;
            interleaved_y = 0;
            LOOP(KERNEL_WIDTH_DIV2, interleaved_y,
            {
                kernel_y = interleaved_y * 2;
                DOT_PRODUCT_16( blockC00, pblockA00[kernel_y    ], pBlockB00[kernel_idx] ); kernel_idx++;
                DOT_PRODUCT_16( blockC00, pblockA00[kernel_y + 1], pBlockB00[kernel_idx] ); kernel_idx++;
                DOT_PRODUCT_16( blockC10, pblockA00[kernel_y    ], pBlockB00[kernel_idx] ); kernel_idx++;
                DOT_PRODUCT_16( blockC10, pblockA00[kernel_y + 1], pBlockB00[kernel_idx] ); kernel_idx++;
            } )
            if ( kernel_width_is_odd )
            {
                kernel_y = interleaved_y * 2;
                DOT_PRODUCT_16( blockC00, pblockA00[kernel_y], pBlockB00[kernel_idx] ); kernel_idx++;
                DOT_PRODUCT_16( blockC10, pblockA00[kernel_y], pBlockB00[kernel_idx] ); kernel_idx++;
            }
        }

        //while( ++patch_row < 1 ); //debug
        while( ++patch_row < KERNEL_HEIGHT );

        src0_read += slice_pitch - ( KERNEL_HEIGHT * ROW_PITCH * DILATION_Y ); // reset to start of next slice of patch
    }
    //while ( ++patch_depth < 1 );  //debug
    while ( ++patch_depth < INPUT_DEPTH );

    // Dst resembles a cube of width x height x (output channel * batches).  Each tile writes:
    // (SIMD * TILE_M) x 1 x TILE_N.  Partial writes most likely generated if padding used.
    int_tp out_offset = global_z * out_pitch_z                                                   // batch offset
     + ( group_x * TILE_N ) * out_pitch_y                                       // channel offset
     + ( ( global_y * TILE_M ) / output_width + OUT_PADDING_HEIGHT) * OUT_PITCH_X  // y offset
     + ( ( global_y * TILE_M ) % output_width ) + OUT_PADDING_LEFT;               // x offset
    __global Dtype *out = dst + out_offset;

    Dtype bias[2];
    Dtype2 *bias_vec;
    bias_vec = (Dtype2*)bias;
    *bias_vec = as_Dtype2(SUB_GROUP_BLOCK_READ2((__global INT_TYPE *)biases + group_x * TILE_N));
    // Work around a potential compiler bug.
    if (group_x > 0xFFFFFFFEul) {
      out[0] = bias[0] + bias[1];
    }
    INTERLEAVED_SIMD16_OUTPUT(dst, out_offset, 0);
}
#endif

#ifdef GEMM_LIKE_CONV_32_2_SIMD16

//////////////////////////////////////////////////////////////////////////////
// Conv_Interleaved_32_2_SIMD16
//
// Convolution: each workitem computes 1 patch x 32 filters worth of output
// data.
#define TILE_M          2
#define TILE_K          KERNEL_WIDTH
#define TILE_N          32

#ifndef __BEIGNET__
__attribute__((intel_reqd_sub_group_size(16)))
#endif
__kernel void Conv_Interleaved(GEMM_LIKE_KERNEL_ARGS)
{
    const int group_x = get_group_id(0);
    const int group_y = get_group_id(1);
    const int global_x = get_global_id(0);
    const int global_y = get_global_id(1);
    const int global_z = get_global_id(2);
    int interleaved_y;
    int kernel_y;
    int kernel_idx;
#define DOT_PRODUCT_16( _result, _rowA, colB )    \
    {   \
        _result.s0 = mad( _rowA, sub_group_broadcast( colB, 0 ), _result.s0 );  \
        _result.s1 = mad( _rowA, sub_group_broadcast( colB, 1 ), _result.s1 );  \
        _result.s2 = mad( _rowA, sub_group_broadcast( colB, 2 ), _result.s2 );  \
        _result.s3 = mad( _rowA, sub_group_broadcast( colB, 3 ), _result.s3 );  \
        _result.s4 = mad( _rowA, sub_group_broadcast( colB, 4 ), _result.s4 );  \
        _result.s5 = mad( _rowA, sub_group_broadcast( colB, 5 ), _result.s5 );  \
        _result.s6 = mad( _rowA, sub_group_broadcast( colB, 6 ), _result.s6 );  \
        _result.s7 = mad( _rowA, sub_group_broadcast( colB, 7 ), _result.s7 );  \
        _result.s8 = mad( _rowA, sub_group_broadcast( colB, 8 ), _result.s8 );  \
        _result.s9 = mad( _rowA, sub_group_broadcast( colB, 9 ), _result.s9 );  \
        _result.sa = mad( _rowA, sub_group_broadcast( colB, 10 ), _result.sa );  \
        _result.sb = mad( _rowA, sub_group_broadcast( colB, 11 ), _result.sb );  \
        _result.sc = mad( _rowA, sub_group_broadcast( colB, 12 ), _result.sc );  \
        _result.sd = mad( _rowA, sub_group_broadcast( colB, 13 ), _result.sd );  \
        _result.se = mad( _rowA, sub_group_broadcast( colB, 14 ), _result.se );  \
        _result.sf = mad( _rowA, sub_group_broadcast( colB, 15 ), _result.sf );  \
    }
        typedef CAT( Dtype, KERNEL_WIDTH ) Dtype_t;

    // True for all threads if filter_width is multiple of TILE_N
    // else, true for all but right-most column of threads.
    {
        // Result ctile (*dst) is M rows x N columns
        // LWG size is 1x8.  Thus each thread calculates 8*M rows x N cols of ctile.
        Dtype16  blockC00 = 0.f;
        Dtype16  blockC10 = 0.f;
        Dtype16  blockC01 = 0.f;
        Dtype16  blockC11 = 0.f;

        // Src0 (patch input) is directly used as atile.
        // Each work item points to the start of a different patch.
        // atile is M rows x K columns.
        int curr_x0 = ( ( global_y * TILE_M + 0 ) % output_width ) * STRIDE_X;
        int curr_x1 = ( ( global_y * TILE_M + 1 ) % output_width ) * STRIDE_X;
        int curr_y0 = ( ( global_y * TILE_M + 0 ) / output_width ) * STRIDE_Y;
        int curr_y1 = ( ( global_y * TILE_M + 1 ) / output_width ) * STRIDE_Y;
#if INPUT_PAD_H != 0 || INPUT_PAD_W != 0 || DILATION_X != 1 || DILATION_Y != 1
        int saved_y0 = curr_y0;
        int saved_y1 = curr_y1;
#endif
        const __global Dtype *src0_read0 = src0
         + aligned_input_size * global_z                                            // batch offset
         + (curr_y0 - INPUT_PAD_H) * ROW_PITCH   // y offset
         + curr_x0 - INPUT_PAD_W;                // x offset
        const __global Dtype *src0_read1 = src0
         + aligned_input_size * global_z                                            // batch offset
         + (curr_y1 - INPUT_PAD_H) * ROW_PITCH   // y offset
         + curr_x1 - INPUT_PAD_W;                // x offset

        // Src1 (filter) is directly used as btile.
        // It starts at the top of src1 and walks down.
        // btile is K rows x N columns.
        const __global Dtype *src1_read = src1 + ( global_x * TILE_N * 2);

        // Walk DOWN src0 (patch 0, 1, 2, ...) and DOWN src1.
        // Inner loop loads and FMADs one row (KERNEL_WIDTH) of each input patch
        // and KERNEL_WIDTH/2 rows of interleaved filter.
        int patch_depth = 0;
        do
        {
            int patch_row = 0;
            do
            {
                // Load atile and btile.
                // Kernel data is partially interleaved.  Every 2 rows are interleaved at Dtype8 granularity.
                // The exception is that if KERNEL_WIDTH is odd the last row is not interleaved.  The non
                // interleaved row is padded with zero to ensure same size as interleaved rows. This
                // interleaving is done to ensure 0% GDR bank conflicts.  For example, this is how the
                // kernel data would be arranged before/after interleaving for KERNEL_WIDTH=3.
                // (0, 0) (8, 0) (16, 0) (24, 0) ...       (0, 0) (0, 1) (8, 0) (0, 1) (16, 0) (0, 1) (24, 0) ..
                // (0, 1) (8, 1) (16, 1) (24, 1) ... =>    (0, 2) (8, 2) (16, 2) (24, 2) ...
                // (0, 2) (8, 2) (16, 2) (24, 2) ...       ...
                // ...
                const bool kernel_width_is_odd = KERNEL_WIDTH % 2 == 1;
#if INPUT_PAD_H == 0 && INPUT_PAD_W == 0 && DILATION_X == 1 && DILATION_Y == 1
                Dtype_t blockA00 = ( (const __global Dtype_t*)src0_read0 )[  0  ]; src0_read0 += ROW_PITCH;
                Dtype_t blockA01 = ( (const __global Dtype_t*)src0_read1 )[  0  ]; src0_read1 += ROW_PITCH;
                Dtype*  pblockA00 = (Dtype*)(&blockA00);
                Dtype*  pblockA01 = (Dtype*)(&blockA01);
#else
                Dtype_t blockA00;
                Dtype*  pblockA00 = (Dtype*)(&blockA00);
                int pos = 0;
                LOOP(KERNEL_WIDTH, pos,
                {
                  if (curr_y0 >= INPUT_PAD_H && curr_y0 < input_height + INPUT_PAD_H && curr_x0 + pos * DILATION_X >= INPUT_PAD_W && curr_x0 + pos * DILATION_X < input_width + INPUT_PAD_W)
                    pblockA00[pos] = src0_read0[pos * DILATION_X];
                  else
                    pblockA00[pos] = 0;
                })
                curr_y0 += DILATION_Y;
                Dtype_t blockA01;
                Dtype*  pblockA01 = (Dtype*)(&blockA01);
                pos = 0;
                LOOP(KERNEL_WIDTH, pos,
                {
                  if (curr_y1 >= INPUT_PAD_H && curr_y1 < input_height + INPUT_PAD_H && curr_x1 + pos * DILATION_X >= INPUT_PAD_W && curr_x1 + pos * DILATION_X < input_width + INPUT_PAD_W)
                    pblockA01[pos] = src0_read1[pos * DILATION_X];
                  else
                    pblockA01[pos] = 0;
                })
                curr_y1 += DILATION_Y;
                src0_read0 += (ROW_PITCH * DILATION_Y);
                src0_read1 += (ROW_PITCH * DILATION_Y);
#endif
                Dtype blockB00[KERNEL_WIDTH*2];
                Dtype4* p4BlockB00 = (Dtype4*)blockB00;
                Dtype2* p2BlockB00 = (Dtype2*)blockB00;
                Dtype*  pBlockB00 =  (Dtype* )blockB00;

                interleaved_y = 0;
                LOOP(KERNEL_WIDTH_DIV2, interleaved_y,
                {
                    p4BlockB00[interleaved_y] = as_Dtype4( SUB_GROUP_BLOCK_READ4( (const __global INT_TYPE*)src1_read ) );
                    src1_read += WIDTH1 * 2;
                } )
                if ( kernel_width_is_odd )
                {
                    p2BlockB00[KERNEL_WIDTH - 1] = as_Dtype2( SUB_GROUP_BLOCK_READ2( (const __global INT_TYPE*)src1_read ) );
                    src1_read += WIDTH1 * 2;
                }
                // Perform MADs
                kernel_idx = 0;
                interleaved_y = 0;
                LOOP(KERNEL_WIDTH_DIV2, interleaved_y,
                {
                    kernel_y = interleaved_y * 2;
                    DOT_PRODUCT_16( blockC00, pblockA00[kernel_y    ], pBlockB00[kernel_idx] );
                    DOT_PRODUCT_16( blockC01, pblockA01[kernel_y    ], pBlockB00[kernel_idx] ); kernel_idx++;
                    DOT_PRODUCT_16( blockC00, pblockA00[kernel_y + 1], pBlockB00[kernel_idx] );
                    DOT_PRODUCT_16( blockC01, pblockA01[kernel_y + 1], pBlockB00[kernel_idx] ); kernel_idx++;
                    DOT_PRODUCT_16( blockC10, pblockA00[kernel_y    ], pBlockB00[kernel_idx] );
                    DOT_PRODUCT_16( blockC11, pblockA01[kernel_y    ], pBlockB00[kernel_idx] ); kernel_idx++;
                    DOT_PRODUCT_16( blockC10, pblockA00[kernel_y + 1], pBlockB00[kernel_idx] );
                    DOT_PRODUCT_16( blockC11, pblockA01[kernel_y + 1], pBlockB00[kernel_idx] ); kernel_idx++;
                } )
                if ( kernel_width_is_odd )
                {
                    kernel_y = interleaved_y * 2;
                    DOT_PRODUCT_16( blockC00, pblockA00[kernel_y], pBlockB00[kernel_idx] );
                    DOT_PRODUCT_16( blockC01, pblockA01[kernel_y], pBlockB00[kernel_idx] ); kernel_idx++;
                    DOT_PRODUCT_16( blockC10, pblockA00[kernel_y], pBlockB00[kernel_idx] );
                    DOT_PRODUCT_16( blockC11, pblockA01[kernel_y], pBlockB00[kernel_idx] ); kernel_idx++;
                }
            }

            //while( ++patch_row < 1 ); //debug
            while( ++patch_row < KERNEL_HEIGHT );
#if INPUT_PAD_W != 0 || INPUT_PAD_H != 0 || DILATION_X != 1 || DILATION_Y != 1
            curr_y0 = saved_y0;
            curr_y1 = saved_y1;
#endif
            src0_read0 += slice_pitch - ( KERNEL_HEIGHT * ROW_PITCH * DILATION_Y); // reset to start of next slice of patch
            src0_read1 += slice_pitch - ( KERNEL_HEIGHT * ROW_PITCH * DILATION_Y);
        }
        //while ( ++patch_depth < 1 );  //debug
        while ( ++patch_depth < INPUT_DEPTH );

        // Dst resembles a cube of width x height x (output channel * batches).  Each tile writes:
        // (SIMD * TILE_M) x 1 x TILE_N.  Partial writes most likely generated if padding used.
        int_tp out0_offset = global_z * out_pitch_z                                                       // batch offset
         + ( group_x * TILE_N ) * out_pitch_y                                           // channel offset
         + ( ( global_y * TILE_M + 0 ) / output_width + OUT_PADDING_HEIGHT ) * OUT_PITCH_X // y offset
         + ( ( global_y * TILE_M + 0 ) % output_width ) + OUT_PADDING_LEFT;               // x offset
        int_tp out1_offset = global_z * out_pitch_z                                                       // batch offset
         + ( group_x * TILE_N ) * out_pitch_y                                           // channel offset
         + ( ( global_y * TILE_M + 1 ) / output_width + OUT_PADDING_HEIGHT ) * OUT_PITCH_X // y offset
         + ( ( global_y * TILE_M + 1 ) % output_width ) + OUT_PADDING_LEFT;               // x offset

        Dtype bias[2];
        Dtype2 *bias_vec;
        bias_vec = (Dtype2*)bias;
        *bias_vec = as_Dtype2(SUB_GROUP_BLOCK_READ2((__global INT_TYPE *)biases + group_x * TILE_N));

        INTERLEAVED_SIMD16_OUTPUT(dst, out0_offset, 0);
        INTERLEAVED_SIMD16_OUTPUT(dst, out1_offset, 1);
    }
}
#endif
