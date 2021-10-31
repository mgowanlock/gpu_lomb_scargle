#include "structs.h"
#include <math.h>

//Only include parameters file if we're not creating the shared library
#ifndef PYTHON
#include "params.h"
#endif

__global__ void lombscarglegpuonethread(DTYPE * x, DTYPE * y, DTYPE * freqs, unsigned int * sizexy, unsigned int * sizefreqs, DTYPE * pgram);
__global__ void lombscargleFindMaxPowers(struct lookupObj * objectLookup, DTYPE * pgram, unsigned int * sizefreqs, DTYPE * periodsList, DTYPE * maxPower);
__forceinline__ __device__ void parReductionMaximumPowerinSM(DTYPE maxPowerForComputingPeriod[], unsigned int maxPowerIdxForComputingPeriod[]);
__global__ void lombscargleBatch(DTYPE * x, DTYPE * y, struct lookupObj * objectLookup, DTYPE * pgram,  DTYPE * foundPeriod, const double minFreq, const double maxFreq, const unsigned int numFreqs);
__global__ void lombscargleBatchSM(DTYPE * x, DTYPE * y, struct lookupObj * objectLookup, DTYPE * pgram,  DTYPE * foundPeriod, const double minFreq, const double maxFreq, const unsigned int numFreqs);
__global__ void lombscargleOneObject(DTYPE * x, DTYPE * y, DTYPE * pgram,  const unsigned int sizeData, const double minFreq, const double maxFreq, const unsigned int numFreqs);
__global__ void lombscargleOneObjectSM(DTYPE * x, DTYPE * y, DTYPE * pgram,  const unsigned int sizeData, const double minFreq, const double maxFreq, const unsigned int numFreqs);

__global__ void lombscargleBatchError(DTYPE * x, DTYPE * y, DTYPE * dy, struct lookupObj * objectLookup, DTYPE * pgram,  DTYPE * foundPeriod, 
	const double minFreq, const double maxFreq, const unsigned int numFreqs);
__global__ void lombscargleOneObjectError(DTYPE * x, DTYPE * y, DTYPE * dy, DTYPE * pgram, const unsigned int sizeData, const double minFreq, const double maxFreq, const unsigned int numFreqs);