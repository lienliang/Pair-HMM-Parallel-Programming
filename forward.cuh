//Author: Enliang Li
//header file for the forward algorithm (gpu-version)
//Latest Version: 1.0 on June.18th 2019

#ifndef _FORWARD_CUH_
#define _FORWARD_CUH_
#include <cstdlib>
#include <iostream>
#include <cmath>
#include <algorithm>
#include <vector>
#include <chrono>
#include <ctime>
#include <ratio>
#include <fstream>
#include <string>
#include <assert.h>

//Define any useful program-wide constants here
#define RSLEN 128  //rslen is ReadLength (maximum allowed 128) => x_pos
#define HAPLEN 128   //haplen is HaplotypeLength (maximum allowed 128) => y_pos
#define BATCH_REG 33
#define BATCH_LAG 22
#define STATES 3

#define matchToMatch 0
#define indelToMatch 1
#define matchToInsertion 2
#define insertionToInsertion 3
#define matchToDeletion 4
#define deletionToDeletion 5

#define MAX_QUAL 254
#define MAX_TOLERANCE 8.0
#define INITIAL_CONDITION pow(2,1020)
#define INITIAL_CONDITION_LOG10 log10(pow(2,1020))
#define CACHE_SIZE (((MAX_QUAL + 1) * (MAX_QUAL + 2)) >> 1)

typedef struct {
  size_t rs_len, hap_len;
  std::vector<uint8_t> q;
  std::vector<uint8_t> i;
  std::vector<uint8_t> d;
  std::vector<uint8_t> c;
  std::vector<uint8_t> hap;
  std::vector<uint8_t> rs;
} testcase;

double approximateLog10SumLog10(
  double a,
  double b
);

double matchToMatchProb(char insQual, char delQual, double matchToMatchProb_Cache[((MAX_QUAL + 1) * (MAX_QUAL + 2)) >> 1]);
double qualToErrorProb(double qual);
double qualToProb(double qual);
void qualToTransProbs(
  double dest[6],
  char insQual,
  char delQual,
  char gcp,
  double matchToMatchProb_Cache[CACHE_SIZE]
);

int subComputeReadLikelihoodGivenHaplotypeLog10(
  double result[BATCH_REG],
  int effective_copy,
  unsigned int effective_rslen[BATCH_REG],
  unsigned int effective_haplen[BATCH_REG],
  unsigned char haplotypeBases[BATCH_REG * HAPLEN],
  unsigned char readBases[BATCH_REG * RSLEN],
  unsigned char readQuals[BATCH_REG * RSLEN],
  unsigned char insertionGOP[BATCH_REG * RSLEN],
  unsigned char deletionGOP[BATCH_REG * RSLEN],
  unsigned char overallGCP[BATCH_REG * RSLEN]
);




#endif
