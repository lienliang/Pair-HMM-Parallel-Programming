//Author: Enliang Li
//diagonal version for the forward algorithm (gpu-diagonal-pipeline-version)
//Latest Version: 4.0 on September.4th 2020

#include "forward.cuh"
#include "auxilliary.cpp"
#include "cuda.h"
#include "cuda_runtime.h"
#include <stdio.h>
#include <assert.h>


#define cdpErrchk(ans) { cdpAssert((ans), __FILE__, __LINE__); }
__device__ void cdpAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      printf("GPU kernel assert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) assert(0);
   }
}

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}


//Define any useful program-wide constants here
#define MAX_THREADS_PER_BLOCK 1024

#define forward_4d(i3, i2, i1, i0) forward_matrix[i3 * (HAPLEN+1) * BATCH_REG * (STATES) + i2 * BATCH_REG * (STATES) + i1 * (STATES) + i0]
/*i3 = rs_pos, i2 = hapl_pos, i1 = batch_id, i0 = states_id*/

#define prior_3d(i2, i1, i0) prior[i2 * (HAPLEN+1) * BATCH_REG  + i1 * BATCH_REG + i0]
/*i2 = rs_pos, i1 = hapl_pos, i0 = batch_id*/

#define transitions_3d(i2, i1, i0) transition[i2 * 6 * (RSLEN+1)  + i1 * (RSLEN+1) + i0]
/*i2 = batch_id, i1 = type_id, i0 = rs_pos*/




/************************ auxilliary functions for GPU *************************/
__device__ double approximateLog10SumLog10_gpu(double a, double b) {
  double diff = 0.0;
  if (a > b) {
    diff = a - b;
  }
  else if (a == -INFINITY) {
    return b;
  }
  else {diff = b - a;}

  double val = 0.0;

  if (diff < MAX_TOLERANCE) {
    val = log10(1.0 + pow(10.0, -diff));
  }

  return val;
}

/**************************** initialization function *****************************/

__global__ void initialization_gpu(double *forward_matrix, double *matchToMatchProb_Cache, unsigned int* effective_rslen, unsigned int *effective_haplen, double *transition, unsigned char *insertionGOP, unsigned char *deletionGOP, unsigned char *overallGCP, double *prior, unsigned char *haplotypeBases, unsigned char *readBases, unsigned char *readQuals) {

  // Part 1:
  /* Operation:
    dim3 dimBlock(MAX_QUAL + 1, 1, 1);
    dim3 dimGrid(1, MAX_QUAL + 1, 1);
  */

  int t = blockIdx.x * MAX_THREADS_PER_BLOCK + threadIdx.x;

  int batch_id = blockIdx.y;
  int x_dim = t / (MAX_QUAL + 1);
  int y_dim = t % (MAX_QUAL + 1);

  if (x_dim < MAX_QUAL + 1) {
    int offset = (((x_dim + 1) * x_dim) >> 1);
  
    if (y_dim <= x_dim) {
      double log10Sum = 0.0;
      log10Sum = approximateLog10SumLog10_gpu(-0.1 * x_dim, -0.1 * y_dim);
      double intermediate = log1p(-min(1.0, pow(10, log10Sum))) * (1.0 / log(10.0));
      matchToMatchProb_Cache[batch_id*CACHE_SIZE+(offset+y_dim)] = pow(10, intermediate);
    }
  }
  __syncthreads();

  // Part 2:
  /* Operation:
    dim3 dimBlock(rslen + 1, 1, 1);
    dim3 dimGrid(1, 6, 1);
  */
  
  int rs_id = t / 6;
  int type_id = t % 6;
  
  if (rs_id < effective_rslen[batch_id] + 1) {
    switch(type_id) 
    {
      case 0:
      {
        int insQual_int = (insertionGOP[batch_id*(RSLEN+1)+rs_id] & 0xFF);
        int delQual_int = (deletionGOP[batch_id*(RSLEN+1)+rs_id] & 0xFF);

        int cur_minQual;
        int cur_maxQual;
        if (insQual_int <= delQual_int) {
          cur_minQual = insQual_int;
          cur_maxQual = delQual_int;
        }
        else {
          cur_minQual = delQual_int;
          cur_maxQual = insQual_int;
        }

        double val = 0.0;
        double temp = 0.0;

        if (cur_maxQual > MAX_QUAL) {
          temp = approximateLog10SumLog10_gpu(-0.1*cur_minQual, -0.1*cur_maxQual);
          val = 1.0 - pow(10, temp);
        }
        else {
          val = matchToMatchProb_Cache[batch_id*CACHE_SIZE+((cur_maxQual * (cur_maxQual + 1)) >> 1) + cur_minQual];
        }

        transitions_3d(batch_id, 0, rs_id + 1) = val;
        break;
      }
      case 1:
      {  
        transitions_3d(batch_id, 1, rs_id+1) = (1.0 - pow(10.0, overallGCP[batch_id*(RSLEN+1)+rs_id] / -10.0));
        break;
      }
      case 2:
      {
        transitions_3d(batch_id, 2, rs_id+1) = pow(10.0, insertionGOP[batch_id*(RSLEN+1)+rs_id] / -10.0);
        break;
      }
      case 3:
      {
        transitions_3d(batch_id, 3, rs_id+1) = pow(10.0, overallGCP[batch_id*(RSLEN+1)+rs_id] / -10.0);
        break;
      }
      case 4:
      { 
        transitions_3d(batch_id, 4, rs_id+1) = pow(10.0, deletionGOP[batch_id*(RSLEN+1)+rs_id] / -10.0);
        break;
      }
      case 5:
      {
        transitions_3d(batch_id, 5, rs_id+1) = pow(10.0, overallGCP[batch_id*(RSLEN+1)+rs_id] / -10.0);
        break;
      }
    }
  }


  __syncthreads();
 
  
  // Part 3:
  /* Operation:
    dim3 dimBlock(haplen, 1, 1);
    dim3 dimGrid(1, rslen, 1);
  */
  
  int i = t / HAPLEN;
  unsigned char x = readBases[i];
  
  unsigned int qual = readQuals[i];
  

  int j = t % HAPLEN;

  if (i < RSLEN) {
    if (j >= 1) {
      unsigned char y = haplotypeBases[j];
      prior_3d((i+1), (j+1), batch_id) =  (x == y || x == 'N' || y == 'N' ) ? (1.0 - pow(10.0, qual / -10.0)) : (pow(10.0, qual / -10.0));  //If we need tristate error, modify here
    }
  }

  __syncthreads();

  // Part 4:
  /* Operation:
    Assign Initial Value to delete state
  */

  if (t < effective_haplen[batch_id]) {
    forward_4d(0, t, batch_id, 2) = INITIAL_CONDITION / effective_haplen[batch_id];
  }

  __syncthreads();
}




//Main function for Pair-HMM in GPU
__global__
void pair_HMM_diagonal(
  unsigned int *effective_rslen,
  unsigned int *effective_haplen,
  double *forward_matrix,
  double *matchToMatchProb_Cache,
  double *transition,
  double *prior,
  double *finalSumProbabilities,
  unsigned char *haplotypeBases,
  unsigned char *readBases,
  unsigned char *readQuals,
  unsigned char *insertionGOP,
  unsigned char *deletionGOP,
  unsigned char *overallGCP
) {


  int batch_id = blockIdx.x;
  int states_id = threadIdx.y;
  int diagonal_id = threadIdx.x;
  int unit_length = blockDim.x;
  int num_batch = gridDim.x;

  /*
    For Ref:
      dim3 dimBlock(states, max_diagonal_len, 1);
      dim3 dimGrid(effective_copy, 1, 1);
  */

  __shared__ int curBlock_configuration[2]; // effective_rslen, effective_haplen


  if (states_id == 0 && diagonal_id == 0) {

    dim3 dimBlock(MAX_THREADS_PER_BLOCK, 1, 1);
    dim3 dimGrid(ceil(HAPLEN * RSLEN / (float)MAX_THREADS_PER_BLOCK), num_batch, 1);
    
    initialization_gpu<<<dimGrid, dimBlock>>>(forward_matrix, matchToMatchProb_Cache, effective_rslen, effective_haplen, transition, insertionGOP, deletionGOP, overallGCP, prior, haplotypeBases, readBases, readQuals);
    
    //printf("Done Initialization\n");
    cudaDeviceSynchronize();


    curBlock_configuration[0] = effective_rslen[batch_id]; //rslen is in x_ditection
    curBlock_configuration[1] = effective_haplen[batch_id]; //haplen is in y_direction
    finalSumProbabilities[batch_id] = 0.0f;

  }


  __syncthreads();

  int x_pos = 0;
  int y_pos = 0;
  
  // First move towards the diagonal
  for (int x_offset = 1; x_offset < curBlock_configuration[0]; x_offset++) {

    for (int offset_factor = 0; offset_factor * unit_length < x_offset; offset_factor++) {
      
      x_pos = (x_offset - offset_factor * unit_length) - diagonal_id;
      y_pos = (1 + offset_factor * unit_length) + diagonal_id;

      if (x_pos > 0 && y_pos < curBlock_configuration[1]) {
      
        //start calculating
        if (states_id == 0) {
          forward_4d(x_pos, y_pos, batch_id, 0) = (prior_3d((x_pos-1), (y_pos-1), batch_id) * (forward_4d((x_pos-1), (y_pos-1), batch_id, 0) * transitions_3d(batch_id, matchToMatch, x_pos) +
                  forward_4d((x_pos-1), (y_pos-1), batch_id, 1) * transitions_3d(batch_id, indelToMatch, x_pos) +
                  forward_4d((x_pos-1), (y_pos-1), batch_id, 2) * transitions_3d(batch_id, indelToMatch, x_pos)));
                  
        } else if (states_id == 1) {
          forward_4d(x_pos, y_pos, batch_id, 1) = (forward_4d((x_pos-1), y_pos, batch_id, 0) * transitions_3d(batch_id, matchToInsertion, x_pos) + forward_4d((x_pos-1), y_pos, batch_id, 1) * transitions_3d(batch_id, insertionToInsertion, x_pos));
          
        } else {
          forward_4d(x_pos, y_pos, batch_id, 2) = (forward_4d(x_pos, (y_pos-1), batch_id, 0) * transitions_3d(batch_id, matchToDeletion, x_pos) + forward_4d(x_pos, (y_pos-1), batch_id, 2) * transitions_3d(batch_id, deletionToDeletion, x_pos));
        }

      }

    }

  }

  
  // Then move towards the end spot of calculation (from diagonal)
  for (int y_offset = 1; y_offset < curBlock_configuration[1]; y_offset++) {

    for (int offset_factor = 0; offset_factor * unit_length < y_offset; offset_factor++) {

      x_pos = (curBlock_configuration[0] - 1 - offset_factor * unit_length - diagonal_id);
      y_pos = (y_offset + offset_factor * unit_length + diagonal_id);
      
      if (x_pos > 0 && y_pos < curBlock_configuration[1]) {
      
        //start calculating
        if (states_id == 0) {
          forward_4d(x_pos, y_pos, batch_id, 0) = (prior_3d((x_pos-1), (y_pos-1), batch_id) * (forward_4d((x_pos-1), (y_pos-1), batch_id, 0) * transitions_3d(batch_id, matchToMatch, x_pos) +
                  forward_4d((x_pos-1), (y_pos-1), batch_id, 1) * transitions_3d(batch_id, indelToMatch, x_pos) +
                  forward_4d((x_pos-1), (y_pos-1), batch_id, 2) * transitions_3d(batch_id, indelToMatch, x_pos)));
                  
        } else if (states_id == 1) {
          forward_4d(x_pos, y_pos, batch_id, 1) = (forward_4d((x_pos-1), y_pos, batch_id, 0) * transitions_3d(batch_id, matchToInsertion, x_pos) + forward_4d((x_pos-1), y_pos, batch_id, 1) * transitions_3d(batch_id, insertionToInsertion, x_pos));
          
        } else {
          forward_4d(x_pos, y_pos, batch_id, 2) = (forward_4d(x_pos, (y_pos-1), batch_id, 0) * transitions_3d(batch_id, matchToDeletion, x_pos) + forward_4d(x_pos, (y_pos-1), batch_id, 2) * transitions_3d(batch_id, deletionToDeletion, x_pos));
        }

      }

    }
    
  }

  
  if (states_id == 0 && diagonal_id == 0) {
    finalSumProbabilities[batch_id] = log10(finalSumProbabilities[batch_id]) - INITIAL_CONDITION_LOG10;
  }

  
  return;
}

#undef transitions_3d
#undef forward_4d
#undef prior_3d




//host function entry is here
int subComputeReadLikelihoodGivenHaplotypeLog10(
  double result[BATCH_REG],
  int effective_copy,
  unsigned int effective_rslen[BATCH_REG],
  unsigned int effective_haplen[BATCH_REG],
  unsigned char haplotypeBases[BATCH_REG*HAPLEN],
  unsigned char readBases[BATCH_REG*RSLEN],
  unsigned char readQuals[BATCH_REG*RSLEN],
  unsigned char insertionGOP[BATCH_REG*RSLEN],
  unsigned char deletionGOP[BATCH_REG*RSLEN],
  unsigned char overallGCP[BATCH_REG*RSLEN]
)
{

  unsigned int *dev_effective_rslen;
  unsigned int *dev_effective_haplen;

  double *dev_cur_forward;
  double *dev_finalSumProbabilities;
  unsigned char *dev_insertionGOP;
  unsigned char *dev_deletionGOP;
  unsigned char *dev_overallGCP;
  unsigned char *dev_haplotypeBases;
  unsigned char *dev_readBases;
  unsigned char *dev_readQuals;

  double *dev_trans;
  double *dev_prior;
  double *matchToMatchProb_Cache;

  //Calculate the size of each array
  size_t forward_matrix_size = (RSLEN+1) * (HAPLEN+1) * effective_copy * STATES * sizeof(double);
  size_t GOP_size = RSLEN * effective_copy * sizeof(unsigned char);
  size_t transition_size = (RSLEN + 1) * effective_copy * 6 * sizeof(double);
  size_t prior_size = (RSLEN+1) * (HAPLEN+1) * effective_copy * STATES * sizeof(double);
  size_t matchToMatchProb_Cache_size = CACHE_SIZE * BATCH_REG * sizeof(double);
  size_t haplotype_size = (HAPLEN) * BATCH_REG * sizeof(unsigned char);
  size_t read_size = (RSLEN) * BATCH_REG * sizeof(unsigned char);


  //Malloc corresponding space for on device calculation
  cudaMalloc((void**)&dev_effective_rslen, effective_copy * sizeof(unsigned int));
  cudaMalloc((void**)&dev_effective_haplen, effective_copy * sizeof(unsigned int));
  cudaMalloc((void**)&dev_finalSumProbabilities, effective_copy * sizeof(double));
  cudaMalloc((void**)&dev_cur_forward, forward_matrix_size);

  cudaMalloc((void**)&dev_insertionGOP, GOP_size);
  cudaMalloc((void**)&dev_deletionGOP, GOP_size);
  cudaMalloc((void**)&dev_overallGCP, GOP_size);
  cudaMalloc((void**)&dev_haplotypeBases, haplotype_size);
  cudaMalloc((void**)&dev_readBases, read_size);
  cudaMalloc((void**)&dev_readQuals, read_size);


  //initialize helper arrays to be zeros
  cudaMalloc((void**)&dev_trans, transition_size);
  cudaMemset(dev_trans, 0.0f, transition_size);
  
  cudaMalloc((void**)&dev_prior, prior_size);
  cudaMemset(dev_prior, 0.0f, prior_size);
  
  cudaMalloc((void**)&matchToMatchProb_Cache, matchToMatchProb_Cache_size);
  cudaMemset(matchToMatchProb_Cache, 0.0f, matchToMatchProb_Cache_size);


  //declare the number of diagonal length
  //int max_diagonal_len = 300;

  //define the Grid and Block dimension
  dim3 dimBlock(64, STATES, 1);
  dim3 dimGrid(effective_copy, 1, 1);

  std::cout << "Start Kernel" << std::endl;
  //Start timer
  auto t1 = std::chrono::high_resolution_clock::now();

  cudaMemcpy(dev_effective_haplen, effective_haplen, effective_copy * sizeof(unsigned int), cudaMemcpyHostToDevice);
  cudaMemcpy(dev_effective_rslen, effective_rslen, effective_copy * sizeof(unsigned int), cudaMemcpyHostToDevice);
  cudaMemcpy(dev_insertionGOP, insertionGOP, GOP_size, cudaMemcpyHostToDevice);
  cudaMemcpy(dev_deletionGOP, deletionGOP, GOP_size, cudaMemcpyHostToDevice);
  cudaMemcpy(dev_overallGCP, overallGCP, GOP_size, cudaMemcpyHostToDevice);
  cudaMemcpy(dev_haplotypeBases, haplotypeBases, haplotype_size, cudaMemcpyHostToDevice);
  cudaMemcpy(dev_readBases, readBases, read_size, cudaMemcpyHostToDevice);
  cudaMemcpy(dev_readQuals, readQuals, read_size, cudaMemcpyHostToDevice);


  pair_HMM_diagonal<<<dimGrid, dimBlock>>>
  (
    dev_effective_rslen,
    dev_effective_haplen,
    dev_cur_forward,
    matchToMatchProb_Cache,
    dev_trans,
    dev_prior,
    dev_finalSumProbabilities,
    dev_haplotypeBases,
    dev_readBases,
    dev_readQuals,
    dev_insertionGOP,
    dev_deletionGOP,
    dev_overallGCP
  );

  cudaDeviceSynchronize();

  cudaMemcpy(result, dev_finalSumProbabilities, effective_copy * sizeof(double), cudaMemcpyDeviceToHost);

  //Stop the timer
  auto t2 = std::chrono::high_resolution_clock::now();
  std::cout << "Copy Back Result" << std::endl;
  std::chrono::duration<double, std::milli> milli = (t2 - t1);
  std::cout << "pair HMM took " <<  milli.count() << " milliseconds\n" ;

  //Free all pre-allocated space
  cudaFree(dev_effective_rslen);
  cudaFree(dev_effective_haplen);
  cudaFree(dev_finalSumProbabilities);
  cudaFree(dev_cur_forward);
  cudaFree(dev_insertionGOP);
  cudaFree(dev_deletionGOP);
  cudaFree(dev_overallGCP);
  cudaFree(dev_haplotypeBases);
  cudaFree(dev_readBases);
  cudaFree(dev_readQuals);
  cudaFree(dev_trans);
  cudaFree(dev_prior);
  cudaFree(matchToMatchProb_Cache);


  return 1;
}
