// Copyright (c) 2020 Michael Gowanlock and Brian Donnelly

// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include "structs.h"
#include <math.h>

//FLOATS
//standard L-S
__global__ void lombscargleBatch_float(float * x, float * y, struct lookupObj * objectLookup, float * pgram,  float * foundPeriod, const double minFreq, const double maxFreq, const unsigned int numFreqs);
__global__ void lombscargleOneObject_float(float * x, float * y, float * pgram,  const unsigned int sizeData, const double minFreq, const double maxFreq, const unsigned int numFreqs);

//L-S with errors on magnitudes and floating mean from SciPy
__global__ void lombscargleBatchError_float(float * x, float * y, float * dy, struct lookupObj * objectLookup, float * pgram,  float * foundPeriod, 	const double minFreq, const double maxFreq, const unsigned int numFreqs);
__global__ void lombscargleOneObjectError_float(float * x, float * y, float * dy, float * pgram, const unsigned int sizeData, const double minFreq, const double maxFreq, const unsigned int numFreqs);

//DOUBLES
//standard L-S
__global__ void lombscargleBatch_double(double * x, double * y, struct lookupObj * objectLookup, double * pgram,  double * foundPeriod, const double minFreq, const double maxFreq, const unsigned int numFreqs);
__global__ void lombscargleOneObject_double(double * x, double * y, double * pgram,  const unsigned int sizeData, const double minFreq, const double maxFreq, const unsigned int numFreqs);

//L-S with errors on magnitudes and floating mean from SciPy
__global__ void lombscargleBatchError_double(double * x, double * y, double * dy, struct lookupObj * objectLookup, double * pgram,  double * foundPeriod, 	const double minFreq, const double maxFreq, const unsigned int numFreqs);
__global__ void lombscargleOneObjectError_double(double * x, double * y, double * dy, double * pgram, const unsigned int sizeData, const double minFreq, const double maxFreq, const unsigned int numFreqs);