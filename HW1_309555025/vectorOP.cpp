#include "PPintrin.h"
// implementation of absSerial(), but it is vectorized using PP intrinsics
void absVector(float *values, float *output, int N)
{
  __pp_vec_float x; // floating point vector register length = VECTOR_Width
  __pp_vec_float result; //temporary floating point vector register
  __pp_vec_float zero = _pp_vset_float(0.f);
  //mask: IsNegative, IsNotNegative
  __pp_mask maskAll, maskIsNegative, maskIsNotNegative; // mask with length = VECTOR_Width

  //  Note: Take a careful look at this loop indexing.  This example
  //  code is not guaranteed to work when (N % VECTOR_WIDTH) != 0.
  //  Why is that the case?
  for (int i = 0; i < N; i += VECTOR_WIDTH)
  {

    // All ones
    maskAll = _pp_init_ones();

    // All zeros
    maskIsNegative = _pp_init_ones(0);

    // Load vector of values from contiguous memory addresses
    _pp_vload_float(x, values + i, maskAll); // x = values[i];

    // Set mask according to predicate
    // if(x<0) maskIsNegative = 1111
    // else maskIsNegative = 0000
    _pp_vlt_float(maskIsNegative, x, zero, maskAll); // if (x < 0) {

    // Execute instruction using mask ("if" clause)
    // 0 - x = -x , stores to result
    // if the IF clause is TRUE, maskIsNegative should be All 1
    _pp_vsub_float(result, zero, x, maskIsNegative); //   output[i] = -x;

    // Inverse maskIsNegative to generate "else" mask
    // just like if x >= 0
    maskIsNotNegative = _pp_mask_not(maskIsNegative); // } else {

    // Execute instruction ("else" clause)
    // load vector register starts from values+i to result
    _pp_vload_float(result, values + i, maskIsNotNegative); //   output[i] = x; }

    // Write results back to memory
    // store calculated result to output
    _pp_vstore_float(output + i, result, maskAll);
  }
}

void clampedExpVector(float *values, int *exponents, float *output, int N)
{
  //
  // PP STUDENTS TODO: Implement your vectorized version of
  // clampedExpSerial() here.
  //
  // Your solution should work for any value of
  // N and VECTOR_WIDTH, not just when VECTOR_WIDTH divides N
  //

  // if input length < vector length, calculate it directly
  if(N < VECTOR_WIDTH){
    for (int i = 0; i < N; i++)
    {
      float x = values[i];
      int y = exponents[i];
      if (y == 0)
      {
        output[i] = 1.f;
      }
      else
      {
        float result = x;
        int count = y - 1;
        while (count > 0)
        {
          result *= x;
          count--;
        }
        if (result > 9.999999f)
        {
          result = 9.999999f;
        }
        output[i] = result;
      }
    }
  }
  // if input length > vector width, remember to cut the last part 
  else{
    for (int i = 0; i < N-(N%VECTOR_WIDTH); i += VECTOR_WIDTH)
    {
      /*******************Declration********************/
       //
      __pp_vec_float x; //float x
      __pp_vec_int y; // int y
      __pp_vec_int count;
      __pp_vec_float result; //result, which is atemporary floating point vector register
      __pp_vec_int zero = _pp_vset_int(0);// zero used to comapred with y
      __pp_vec_int one_i = _pp_vset_int(1);
      __pp_vec_float one_f = _pp_vset_float(1.f);
      __pp_mask mask_All, mask_Y_EqualZero, mask_Y_NotEqualZero, mask_IsClamp, mask_NotCountOver; // mask with length = VECTOR_Width


      /*******************Initialization********************/
      // All ones
      mask_All = _pp_init_ones();
      // All zeros
      mask_Y_EqualZero = _pp_init_ones(0);
      mask_Y_NotEqualZero = _pp_init_ones(0);
      mask_IsClamp = _pp_init_ones(0);
      /*******************Main part********************/
      // x, y 
      _pp_vload_float(x, values+i, mask_All);
      _pp_vload_int(y, exponents+i, mask_All);
      // check if exponent = 0
      for(int j = 0; j < VECTOR_WIDTH;j++){
        if(y.value[j] == 0)
          mask_Y_EqualZero.value[j] = true;
        else{
          mask_Y_EqualZero.value[j] = false;
        }
      }
      // ouput = 1.f if exponent = 0
      _pp_vstore_float(output+i, one_f, mask_Y_EqualZero);
    
      // if exponent != 0
      //float result = x
      _pp_vmove_float(result, x, mask_All);
      _pp_vsub_int(count, y, one_i, mask_All);

      // initial mask_IsCountOver
      for(int j = 0; j < VECTOR_WIDTH;j++){
        if(count.value[j] <= 0){
          mask_NotCountOver.value[j] = false;
        }
        else{
          mask_NotCountOver.value[j] = true;
        }
      }
      // while(count > 0)
      while(_pp_cntbits(mask_NotCountOver) > 0){
        // result *= x
        _pp_vmult_float(result, result, x, mask_NotCountOver);
        // count--
        _pp_vsub_int(count, count, one_i, mask_NotCountOver);
        // update mask_IsCountOver
        for(int j = 0; j < VECTOR_WIDTH;j++){
          if(count.value[j] <= 0){
            mask_NotCountOver.value[j] = false;
          }
        }
      }

      // clamp value
      for(int j = 0;j < VECTOR_WIDTH; j++){
        if(result.value[j] > 9.999999f)
          result.value[j] = 9.999999f;
      }
      // create mask_Y_NotEqualZero
      mask_Y_NotEqualZero = _pp_mask_not(mask_Y_EqualZero);
      // output = result;
      _pp_vstore_float(output+i, result, mask_Y_NotEqualZero);
    } // end for(i)  

    // Process last part
    for (int i = N-(N%VECTOR_WIDTH); i < N; i++)
    {
      float x = values[i];
      int y = exponents[i];
      if (y == 0)
      {
        output[i] = 1.f;
      }
      else
      {
        float result = x;
        int count = y - 1;
        while (count > 0)
        {
          result *= x;
          count--;
        }
        if (result > 9.999999f)
        {
          result = 9.999999f;
        }
        output[i] = result;
      }
    }
  } //end else
}// end function

// returns the sum of all elements in values
// You can assume N is a multiple of VECTOR_WIDTH
// You can assume VECTOR_WIDTH is a power of 2
float arraySumVector(float *values, int N)
{

  //
  // PP STUDENTS TODO: Implement your vectorized version of arraySumSerial here
  //

  // Declaration
  __pp_vec_float sum;
  __pp_vec_float val;
  __pp_mask mask_All;
  // All ones
  mask_All = _pp_init_ones();
  // sum = 0
  _pp_vset_float(sum, 0.f, mask_All);
  for (int i = 0; i < N; i += VECTOR_WIDTH)
  {
    _pp_vload_float(val, values+i, mask_All);
    _pp_vadd_float(sum, sum, val, mask_All);
  }
  _pp_hadd_float(sum, sum);
  float cnt = (float)VECTOR_WIDTH;
  cnt /= 2.0;
  while(cnt > 1){
    _pp_interleave_float(sum, sum);
    _pp_hadd_float(sum, sum);
    cnt /= 2.0;
  }
  return sum.value[0];
}