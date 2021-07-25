#include "forward.cuh"

//Define any useful program-wide constants here

using namespace std;

double approximateLog10SumLog10(
  double a,
  double b
){
  if (a > b) {
    return approximateLog10SumLog10(b, a);
  }
  else if (a == -INFINITY) {
    return b;
  }

  double diff = b - a;
  double val = 0.0;

  if (diff < MAX_TOLERANCE) {
    val = log10(1.0 + pow(10.0, -diff));
  }

  return val;
}


double matchToMatchProb(char  insQual, char  delQual, double matchToMatchProb_Cache[((MAX_QUAL + 1) * (MAX_QUAL + 2)) >> 1]) {

  int insQual_int = (insQual & 0xFF);
  int delQual_int = (delQual & 0xFF);

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

  if (cur_maxQual > MAX_QUAL) {
    val = 1.0 - pow(10, approximateLog10SumLog10(-0.1*cur_minQual, -0.1*cur_maxQual));
  }
  else {
    val = matchToMatchProb_Cache[((cur_maxQual * (cur_maxQual + 1)) >> 1) + cur_minQual];
  }

  return val;
}

double qualToErrorProb(double qual) {
  return pow(10.0, qual / -10.0);
}

double qualToProb(double qual) {
  return 1.0 - qualToErrorProb(qual);
}

void qualToTransProbs(
  double dest[6],
  char  insQual,
  char  delQual,
  char  gcp,
  double matchToMatchProb_Cache[CACHE_SIZE]
) {
  dest[0] = matchToMatchProb(insQual, delQual, matchToMatchProb_Cache);
  dest[2] = qualToErrorProb( (double) insQual );
  dest[4] = qualToErrorProb( (double) delQual );
  dest[1] = qualToProb( (double) gcp);
  dest[3] = qualToErrorProb( (double) gcp );
  dest[5] = qualToErrorProb( (double) gcp );
}
