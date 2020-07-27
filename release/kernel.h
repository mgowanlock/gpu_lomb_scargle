#include "structs.h"
#include "params.h"
#include <math.h>

__global__ void lombscargleBatch(DTYPE * x, DTYPE * y, struct lookupObj * objectLookup, DTYPE * pgram,  DTYPE * foundPeriod, const double minFreq, const double maxFreq, const unsigned int numFreqs);
__global__ void lombscargleOneObject(DTYPE * x, DTYPE * y, DTYPE * pgram,  const unsigned int sizeData, const double minFreq, const double maxFreq, const unsigned int numFreqs);
