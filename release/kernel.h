#include "structs.h"
#include "params.h"
#include <math.h>

//standard L-S
__global__ void lombscargleBatch(DTYPE * x, DTYPE * y, struct lookupObj * objectLookup, DTYPE * pgram,  DTYPE * foundPeriod, const double minFreq, const double maxFreq, const unsigned int numFreqs);
__global__ void lombscargleOneObject(DTYPE * x, DTYPE * y, DTYPE * pgram,  const unsigned int sizeData, const double minFreq, const double maxFreq, const unsigned int numFreqs);

//L-S with errors on magnitudes and floating mean from SciPy
__global__ void lombscargleBatchError(DTYPE * x, DTYPE * y, DTYPE * dy, struct lookupObj * objectLookup, DTYPE * pgram,  DTYPE * foundPeriod, 	const double minFreq, const double maxFreq, const unsigned int numFreqs);
__global__ void lombscargleOneObjectError(DTYPE * x, DTYPE * y, DTYPE * dy, DTYPE * pgram, const unsigned int sizeData, const double minFreq, const double maxFreq, const unsigned int numFreqs);