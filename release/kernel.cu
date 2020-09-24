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

#include "kernel.h"

//FLOAT SECTION

//Each thread stores the powers of the pgram and global memory and then have another kernel find the maximum index corresponding to the correct power
__global__ void lombscargleOneObject_float(float * x, float * y, float * pgram, const unsigned int sizeData, const double minFreq, const double maxFreq, const unsigned int numFreqs)
{

	unsigned int tid=(blockIdx.x*blockDim.x)+threadIdx.x; 	
	
	//Values shared by all threads in the block
	__shared__ float freqStep; //increment the frequency by this much

	//Set the values in shared memory using thread 0
	if (threadIdx.x==0)
	{
	freqStep=(maxFreq-minFreq)/(numFreqs*1.0);	
	}
	
	__syncthreads();


    	float c, s;
	    float tau, c_tau, s_tau, c_tau2, s_tau2, cs_tau;
		
			if (tid<numFreqs)
			{


			float freqToTest=minFreq+(freqStep*tid);

	        float xc = 0.0;
	        float xs = 0.0;
	        float cc = 0.0;
	        float ss = 0.0;
	        float cs = 0.0;

	
	        #pragma unroll
	        for (int j=0; j<sizeData; j++)
	        {	
	            // c = cos(freqs[tid] * x[j]);
	            // s = sin(freqs[tid] * x[j]);
	            sincos(freqToTest * x[j], &s, &c); //compute both at same time, see CUDA documentation

	            xc += y[j] * c;
	            xs += y[j] * s;
	            cc += c * c;
	            ss += s * s;
	            cs += c * s;
	        }
	        
	        tau = atan2f(2.0 * cs, cc - ss) / (2.0 * freqToTest);


	        sincos(freqToTest * tau, &s_tau, &c_tau);	       

	        c_tau2 = c_tau * c_tau;
	        s_tau2 = s_tau * s_tau;
	        cs_tau = 2.0 * c_tau * s_tau;

	        

	        // float power=0.5 * (((c_tau * xc + s_tau * xs)*(c_tau * xc + s_tau * xs) / \
	        //     (c_tau2 * cc + cs_tau * cs + s_tau2 * ss)) + \
	        //     ((c_tau * xs - s_tau * xc)*(c_tau * xs - s_tau * xc) / \
	        //     (c_tau2 * ss - cs_tau * cs + s_tau2 * cc)));
	        
	        float f1=(c_tau * xc + s_tau * xs);
	        float f2=(c_tau * xs - s_tau * xc);
	        float d1=(f1*f1);
	        float d2=(c_tau2 * cc + cs_tau * cs + s_tau2 * ss);
	        float d4=(f2*f2);
	        float d5=(c_tau2 * ss - cs_tau * cs + s_tau2 * cc);

	        pgram[tid]=0.5 * ((d1/d2)+(d4/d5));

	    	}	
}















//Compute frequency on the fly in the kernel gievn frequency ranges
__global__ void lombscargleBatch_float(float * x, float * y, struct lookupObj * objectLookup, float * pgram,  float * foundPeriod, 
	const double minFreq, const double maxFreq, const unsigned int numFreqs)
{
	
	
	
	//Values shared by all threads in the block
	__shared__ float freqStep; //increment the frequency by this much

	__shared__ unsigned int minDataIdx;
	__shared__ unsigned int maxDataIdx;
	
	
	__shared__ unsigned int periodWriteOffset;
	

	
	

	//Set the values in shared memory using thread 0
	if (threadIdx.x==0)
	{
		freqStep=(maxFreq-minFreq)/(numFreqs*1.0);	
		minDataIdx=objectLookup[blockIdx.x].idxMin;	
		maxDataIdx=objectLookup[blockIdx.x].idxMax;
		periodWriteOffset=blockIdx.x*numFreqs;
	}
	
	__syncthreads();


    	float c, s;
	    float tau, c_tau, s_tau, c_tau2, s_tau2, cs_tau;
	

		for (int i=0; i<numFreqs && ((i+threadIdx.x)<numFreqs); i+=blockDim.x)
		{
			int idx=i+threadIdx.x;


			float freqToTest=minFreq+(freqStep*idx);

	        float xc = 0.0;
	        float xs = 0.0;
	        float cc = 0.0;
	        float ss = 0.0;
	        float cs = 0.0;

	        //ILP?
	        #pragma unroll
	        for (int j=minDataIdx; j<=maxDataIdx; j++)
	        {	
	            // c = cos(freqs[j] * x[j]);
	            // s = sin(freqs[j] * x[j]);
	        	
	            sincos(freqToTest * x[j], &s, &c);


	            xc += y[j] * c;
	            xs += y[j] * s;
	            cc += c * c;
	            ss += s * s;
	            cs += c * s;
	        }
	        

	        tau = atan2f(2.0 * cs, cc - ss) / (2.0 * freqToTest);


	        
	        sincos(freqToTest * tau, &s_tau, &c_tau);	       
	        
	        

	        c_tau2 = c_tau * c_tau;
	        s_tau2 = s_tau * s_tau;
	        cs_tau = 2.0 * c_tau * s_tau;

	        

	        // float power=0.5 * (((c_tau * xc + s_tau * xs)*(c_tau * xc + s_tau * xs) / \
	        //     (c_tau2 * cc + cs_tau * cs + s_tau2 * ss)) + \
	        //     ((c_tau * xs - s_tau * xc)*(c_tau * xs - s_tau * xc) / \
	        //     (c_tau2 * ss - cs_tau * cs + s_tau2 * cc)));
	        
	        float f1=(c_tau * xc + s_tau * xs);
	        float f2=(c_tau * xs - s_tau * xc);
	        float d1=(f1*f1);
	        float d2=(c_tau2 * cc + cs_tau * cs + s_tau2 * ss);
	        float d4=(f2*f2);
	        float d5=(c_tau2 * ss - cs_tau * cs + s_tau2 * cc);

	        float power=0.5 * ((d1/d2)+(d4/d5));
	        pgram[idx+periodWriteOffset] = power;
	    }
	
}








//The kernel with error only uses global memory and returns the pgram
//generalized L-S from AstroPy
__global__ void lombscargleBatchError_float(float * x, float * y, float * dy, struct lookupObj * objectLookup, float * pgram,  float * foundPeriod, 
	const double minFreq, const double maxFreq, const unsigned int numFreqs)
{
	
	
	
	//Values shared by all threads in the block
	__shared__ float freqStep; //increment the frequency by this much
	__shared__ unsigned int minDataIdx;
	__shared__ unsigned int maxDataIdx;
	__shared__ unsigned int periodWriteOffset;

	//Set the values in shared memory using thread 0
	if (threadIdx.x==0)
	{
		freqStep=(maxFreq-minFreq)/(numFreqs*1.0);	
		minDataIdx=objectLookup[blockIdx.x].idxMin;	
		maxDataIdx=objectLookup[blockIdx.x].idxMax;
		periodWriteOffset=blockIdx.x*numFreqs;
	}
	
	__syncthreads();


	float w, sin_omega_t, cos_omega_t, S, C, S2, C2, tau, Y, wsum, YY, Stau, Ctau, YCtau, YStau, CCtau, SStau; 



	for (int i=0; i<numFreqs && ((i+threadIdx.x)<numFreqs); i+=blockDim.x)
	{


	int idx=i+threadIdx.x;
	float freqToTest=minFreq+(freqStep*idx);

	wsum = 0.0;
	S = 0.0;
	C = 0.0;
	S2 = 0.0;
	C2 = 0.0;

	//first pass: determine tau
	#pragma unroll
	for (int j=minDataIdx; j<=maxDataIdx; j++)
	{
    w = 1.0 / dy[j];
    w *= w;
    wsum += w;
    sincos(freqToTest * x[j], &sin_omega_t, &cos_omega_t);
    S += w * sin_omega_t;
    C += w * cos_omega_t;
    S2 += 2.0 * w * sin_omega_t * cos_omega_t;
    C2 += w - 2.0 * w * sin_omega_t * sin_omega_t;
	}	    

		S2 /= wsum;
		C2 /= wsum;
		S /= wsum;
		C /= wsum;
		S2 -= (2.0 * S * C);
		C2 -= (C * C - S * S);

        tau = 0.5 * atan2f(S2, C2) / freqToTest;

		
		Y = 0.0;
		YY = 0.0;
		Stau = 0.0;
		Ctau = 0.0;
		YCtau = 0.0;
		YStau = 0.0;
		CCtau = 0.0;
		SStau = 0.0;
		// second pass: compute the power
		#pragma unroll
		for (int j=minDataIdx; j<=maxDataIdx; j++)
		{
		    w = 1.0 / dy[j];
		    w *= w;
		    sincos(freqToTest * (x[j] - tau), &sin_omega_t, &cos_omega_t);
		    Y += w * y[j];
		    YY += w * y[j] * y[j];
		    Ctau += w * cos_omega_t;
		    Stau += w * sin_omega_t;
		    YCtau += w * y[j] * cos_omega_t;
		    YStau += w * y[j] * sin_omega_t;
		    CCtau += w * cos_omega_t * cos_omega_t;
		    SStau += w * sin_omega_t * sin_omega_t;
		}
		Y /= wsum;
		YY /= wsum;
		Ctau /= wsum;
		Stau /= wsum;
		YCtau /= wsum;
		YStau /= wsum;
		CCtau /= wsum;
		SStau /= wsum;
		YCtau -= Y * Ctau;
		YStau -= Y * Stau;
		CCtau -= Ctau * Ctau;
		SStau -= Stau * Stau;
		YY -= Y * Y;
		

		pgram[idx+periodWriteOffset] = (YCtau * YCtau / CCtau + YStau * YStau / SStau) / YY;
	}
    
}



//Each thread stores the powers of the pgram and global memory and then have another kernel find the maximum index corresponding to the correct power
__global__ void lombscargleOneObjectError_float(float * x, float * y, float * dy, float * pgram, const unsigned int sizeData, const double minFreq, const double maxFreq, const unsigned int numFreqs)
{

	unsigned int tid=(blockIdx.x*blockDim.x)+threadIdx.x; 	
	
	//Values shared by all threads in the block
	__shared__ float freqStep; //increment the frequency by this much

	//Set the values in shared memory using thread 0
	if (threadIdx.x==0)
	{
		freqStep=(maxFreq-minFreq)/(numFreqs*1.0);	
	}
	
	__syncthreads();


		float w, sin_omega_t, cos_omega_t, S, C, S2, C2, tau, Y, wsum, YY, Stau, Ctau, YCtau, YStau, CCtau, SStau; 
		
		wsum = 0.0;
		S = 0.0;
		C = 0.0;
		S2 = 0.0;
		C2 = 0.0;

		if (tid<numFreqs)
		{

		float freqToTest=minFreq+(freqStep*tid);	
    
       	//first pass: determine tau
       	#pragma unroll
		for (int j=0; j<sizeData; j++)
		{
	    w = 1.0 / dy[j];
	    w *= w;
	    wsum += w;
	    sincos(freqToTest * x[j], &sin_omega_t, &cos_omega_t);
	    S += w * sin_omega_t;
	    C += w * cos_omega_t;
	    S2 += 2.0 * w * sin_omega_t * cos_omega_t;
	    C2 += w - 2.0 * w * sin_omega_t * sin_omega_t;
		}

		S2 /= wsum;
		C2 /= wsum;
		S /= wsum;
		C /= wsum;
		S2 -= (2.0 * S * C);
		C2 -= (C * C - S * S);

		tau = 0.5 * atan2f(S2, C2) / freqToTest;

		Y = 0.0;
		YY = 0.0;
		Stau = 0.0;
		Ctau = 0.0;
		YCtau = 0.0;
		YStau = 0.0;
		CCtau = 0.0;
		SStau = 0.0;
		// second pass: compute the power
		#pragma unroll
		for (int j=0; j<sizeData; j++)
		{
		    w = 1.0 / dy[j];
		    w *= w;
		    sincos(freqToTest * (x[j] - tau), &sin_omega_t, &cos_omega_t);
		    Y += w * y[j];
		    YY += w * y[j] * y[j];
		    Ctau += w * cos_omega_t;
		    Stau += w * sin_omega_t;
		    YCtau += w * y[j] * cos_omega_t;
		    YStau += w * y[j] * sin_omega_t;
		    CCtau += w * cos_omega_t * cos_omega_t;
		    SStau += w * sin_omega_t * sin_omega_t;
		}
		Y /= wsum;
		YY /= wsum;
		Ctau /= wsum;
		Stau /= wsum;
		YCtau /= wsum;
		YStau /= wsum;
		CCtau /= wsum;
		SStau /= wsum;
		YCtau -= Y * Ctau;
		YStau -= Y * Stau;
		CCtau -= Ctau * Ctau;
		SStau -= Stau * Stau;
		YY -= Y * Y;
		


    	pgram[tid] = (YCtau * YCtau / CCtau + YStau * YStau / SStau) / YY;	    



	    }	
}

//DOUBLES SECTION


//Each thread stores the powers of the pgram and global memory and then have another kernel find the maximum index corresponding to the correct power
__global__ void lombscargleOneObject_double(double * x, double * y, double * pgram, const unsigned int sizeData, const double minFreq, const double maxFreq, const unsigned int numFreqs)
{

	unsigned int tid=(blockIdx.x*blockDim.x)+threadIdx.x; 	
	
	//Values shared by all threads in the block
	__shared__ double freqStep; //increment the frequency by this much

	//Set the values in shared memory using thread 0
	if (threadIdx.x==0)
	{
		freqStep=(maxFreq-minFreq)/(numFreqs*1.0);	
	}
	
	__syncthreads();


    	double c, s;
	    double tau, c_tau, s_tau, c_tau2, s_tau2, cs_tau;
		
			if (tid<numFreqs)
			{


			double freqToTest=minFreq+(freqStep*tid);

	        double xc = 0.0;
	        double xs = 0.0;
	        double cc = 0.0;
	        double ss = 0.0;
	        double cs = 0.0;

	
	        #pragma unroll
	        for (int j=0; j<sizeData; j++)
	        {	
	            // c = cos(freqs[tid] * x[j]);
	            // s = sin(freqs[tid] * x[j]);
	            sincos(freqToTest * x[j], &s, &c); //compute both at same time, see CUDA documentation

	            xc += y[j] * c;
	            xs += y[j] * s;
	            cc += c * c;
	            ss += s * s;
	            cs += c * s;
	        }
	        
	        tau = atan2(2.0 * cs, cc - ss) / (2.0 * freqToTest);


	        sincos(freqToTest * tau, &s_tau, &c_tau);	       

	        c_tau2 = c_tau * c_tau;
	        s_tau2 = s_tau * s_tau;
	        cs_tau = 2.0 * c_tau * s_tau;

	        

	        // double power=0.5 * (((c_tau * xc + s_tau * xs)*(c_tau * xc + s_tau * xs) / \
	        //     (c_tau2 * cc + cs_tau * cs + s_tau2 * ss)) + \
	        //     ((c_tau * xs - s_tau * xc)*(c_tau * xs - s_tau * xc) / \
	        //     (c_tau2 * ss - cs_tau * cs + s_tau2 * cc)));
	        
	        double f1=(c_tau * xc + s_tau * xs);
	        double f2=(c_tau * xs - s_tau * xc);
	        double d1=(f1*f1);
	        double d2=(c_tau2 * cc + cs_tau * cs + s_tau2 * ss);
	        double d4=(f2*f2);
	        double d5=(c_tau2 * ss - cs_tau * cs + s_tau2 * cc);

	        pgram[tid]=0.5 * ((d1/d2)+(d4/d5));

	    	}	
}















//Compute frequency on the fly in the kernel gievn frequency ranges
__global__ void lombscargleBatch_double(double * x, double * y, struct lookupObj * objectLookup, double * pgram,  double * foundPeriod, 
	const double minFreq, const double maxFreq, const unsigned int numFreqs)
{
	
	
	
	//Values shared by all threads in the block
	__shared__ double freqStep; //increment the frequency by this much

	__shared__ unsigned int minDataIdx;
	__shared__ unsigned int maxDataIdx;
	
	
	__shared__ unsigned int periodWriteOffset;
	

	
	

	//Set the values in shared memory using thread 0
	if (threadIdx.x==0)
	{
	freqStep=(maxFreq-minFreq)/(numFreqs*1.0);	
	minDataIdx=objectLookup[blockIdx.x].idxMin;	
	maxDataIdx=objectLookup[blockIdx.x].idxMax;
	periodWriteOffset=blockIdx.x*numFreqs;
	}
	
	__syncthreads();


    	double c, s;
	    double tau, c_tau, s_tau, c_tau2, s_tau2, cs_tau;
	

		for (int i=0; i<numFreqs && ((i+threadIdx.x)<numFreqs); i+=blockDim.x)
		{
			int idx=i+threadIdx.x;


			double freqToTest=minFreq+(freqStep*idx);

	        double xc = 0.0;
	        double xs = 0.0;
	        double cc = 0.0;
	        double ss = 0.0;
	        double cs = 0.0;

	        //ILP?
	        #pragma unroll
	        for (int j=minDataIdx; j<=maxDataIdx; j++)
	        {	
	            // c = cos(freqs[j] * x[j]);
	            // s = sin(freqs[j] * x[j]);
	        	
	            sincos(freqToTest * x[j], &s, &c);


	            xc += y[j] * c;
	            xs += y[j] * s;
	            cc += c * c;
	            ss += s * s;
	            cs += c * s;
	        }
	        

	        tau = atan2(2.0 * cs, cc - ss) / (2.0 * freqToTest);


	        
	        sincos(freqToTest * tau, &s_tau, &c_tau);	       
	        
	        

	        c_tau2 = c_tau * c_tau;
	        s_tau2 = s_tau * s_tau;
	        cs_tau = 2.0 * c_tau * s_tau;

	        

	        // double power=0.5 * (((c_tau * xc + s_tau * xs)*(c_tau * xc + s_tau * xs) / \
	        //     (c_tau2 * cc + cs_tau * cs + s_tau2 * ss)) + \
	        //     ((c_tau * xs - s_tau * xc)*(c_tau * xs - s_tau * xc) / \
	        //     (c_tau2 * ss - cs_tau * cs + s_tau2 * cc)));
	        
	        double f1=(c_tau * xc + s_tau * xs);
	        double f2=(c_tau * xs - s_tau * xc);
	        double d1=(f1*f1);
	        double d2=(c_tau2 * cc + cs_tau * cs + s_tau2 * ss);
	        double d4=(f2*f2);
	        double d5=(c_tau2 * ss - cs_tau * cs + s_tau2 * cc);

	        double power=0.5 * ((d1/d2)+(d4/d5));
	        pgram[idx+periodWriteOffset] = power;
	    }
	
}








//The kernel with error only uses global memory and returns the pgram
//generalized L-S from AstroPy
__global__ void lombscargleBatchError_double(double * x, double * y, double * dy, struct lookupObj * objectLookup, double * pgram,  double * foundPeriod, 
	const double minFreq, const double maxFreq, const unsigned int numFreqs)
{
	
	
	
	//Values shared by all threads in the block
	__shared__ double freqStep; //increment the frequency by this much
	__shared__ unsigned int minDataIdx;
	__shared__ unsigned int maxDataIdx;
	__shared__ unsigned int periodWriteOffset;

	//Set the values in shared memory using thread 0
	if (threadIdx.x==0)
	{
	freqStep=(maxFreq-minFreq)/(numFreqs*1.0);	
	minDataIdx=objectLookup[blockIdx.x].idxMin;	
	maxDataIdx=objectLookup[blockIdx.x].idxMax;
	periodWriteOffset=blockIdx.x*numFreqs;
	}
	
	__syncthreads();


	double w, sin_omega_t, cos_omega_t, S, C, S2, C2, tau, Y, wsum, YY, Stau, Ctau, YCtau, YStau, CCtau, SStau; 



	for (int i=0; i<numFreqs && ((i+threadIdx.x)<numFreqs); i+=blockDim.x)
	{


	int idx=i+threadIdx.x;
	double freqToTest=minFreq+(freqStep*idx);

	wsum = 0.0;
	S = 0.0;
	C = 0.0;
	S2 = 0.0;
	C2 = 0.0;

	//first pass: determine tau
	#pragma unroll
	for (int j=minDataIdx; j<=maxDataIdx; j++)
	{
    w = 1.0 / dy[j];
    w *= w;
    wsum += w;
    sincos(freqToTest * x[j], &sin_omega_t, &cos_omega_t);
    S += w * sin_omega_t;
    C += w * cos_omega_t;
    S2 += 2.0 * w * sin_omega_t * cos_omega_t;
    C2 += w - 2.0 * w * sin_omega_t * sin_omega_t;
	}	    

		S2 /= wsum;
		C2 /= wsum;
		S /= wsum;
		C /= wsum;
		S2 -= (2.0 * S * C);
		C2 -= (C * C - S * S);

        tau = 0.5 * atan2(S2, C2) / freqToTest;

		
		Y = 0.0;
		YY = 0.0;
		Stau = 0.0;
		Ctau = 0.0;
		YCtau = 0.0;
		YStau = 0.0;
		CCtau = 0.0;
		SStau = 0.0;
		// second pass: compute the power
		#pragma unroll
		for (int j=minDataIdx; j<=maxDataIdx; j++)
		{
		    w = 1.0 / dy[j];
		    w *= w;
		    sincos(freqToTest * (x[j] - tau), &sin_omega_t, &cos_omega_t);
		    Y += w * y[j];
		    YY += w * y[j] * y[j];
		    Ctau += w * cos_omega_t;
		    Stau += w * sin_omega_t;
		    YCtau += w * y[j] * cos_omega_t;
		    YStau += w * y[j] * sin_omega_t;
		    CCtau += w * cos_omega_t * cos_omega_t;
		    SStau += w * sin_omega_t * sin_omega_t;
		}
		Y /= wsum;
		YY /= wsum;
		Ctau /= wsum;
		Stau /= wsum;
		YCtau /= wsum;
		YStau /= wsum;
		CCtau /= wsum;
		SStau /= wsum;
		YCtau -= Y * Ctau;
		YStau -= Y * Stau;
		CCtau -= Ctau * Ctau;
		SStau -= Stau * Stau;
		YY -= Y * Y;
		

		pgram[idx+periodWriteOffset] = (YCtau * YCtau / CCtau + YStau * YStau / SStau) / YY;
	}
    
}



//Each thread stores the powers of the pgram and global memory and then have another kernel find the maximum index corresponding to the correct power
__global__ void lombscargleOneObjectError_double(double * x, double * y, double * dy, double * pgram, const unsigned int sizeData, const double minFreq, const double maxFreq, const unsigned int numFreqs)
{

	unsigned int tid=(blockIdx.x*blockDim.x)+threadIdx.x; 	
	
	//Values shared by all threads in the block
	__shared__ double freqStep; //increment the frequency by this much

	//Set the values in shared memory using thread 0
	if (threadIdx.x==0)
	{
	freqStep=(maxFreq-minFreq)/(numFreqs*1.0);	
	}
	
	__syncthreads();


		double w, sin_omega_t, cos_omega_t, S, C, S2, C2, tau, Y, wsum, YY, Stau, Ctau, YCtau, YStau, CCtau, SStau; 
		
		wsum = 0.0;
		S = 0.0;
		C = 0.0;
		S2 = 0.0;
		C2 = 0.0;

		if (tid<numFreqs)
		{

		double freqToTest=minFreq+(freqStep*tid);	
    
       	//first pass: determine tau
       	#pragma unroll
		for (int j=0; j<sizeData; j++)
		{
	    w = 1.0 / dy[j];
	    w *= w;
	    wsum += w;
	    sincos(freqToTest * x[j], &sin_omega_t, &cos_omega_t);
	    S += w * sin_omega_t;
	    C += w * cos_omega_t;
	    S2 += 2.0 * w * sin_omega_t * cos_omega_t;
	    C2 += w - 2.0 * w * sin_omega_t * sin_omega_t;
		}

		S2 /= wsum;
		C2 /= wsum;
		S /= wsum;
		C /= wsum;
		S2 -= (2.0 * S * C);
		C2 -= (C * C - S * S);

		tau = 0.5 * atan2(S2, C2) / freqToTest;

		Y = 0.0;
		YY = 0.0;
		Stau = 0.0;
		Ctau = 0.0;
		YCtau = 0.0;
		YStau = 0.0;
		CCtau = 0.0;
		SStau = 0.0;
		// second pass: compute the power
		#pragma unroll
		for (int j=0; j<sizeData; j++)
		{
		    w = 1.0 / dy[j];
		    w *= w;
		    sincos(freqToTest * (x[j] - tau), &sin_omega_t, &cos_omega_t);
		    Y += w * y[j];
		    YY += w * y[j] * y[j];
		    Ctau += w * cos_omega_t;
		    Stau += w * sin_omega_t;
		    YCtau += w * y[j] * cos_omega_t;
		    YStau += w * y[j] * sin_omega_t;
		    CCtau += w * cos_omega_t * cos_omega_t;
		    SStau += w * sin_omega_t * sin_omega_t;
		}
		Y /= wsum;
		YY /= wsum;
		Ctau /= wsum;
		Stau /= wsum;
		YCtau /= wsum;
		YStau /= wsum;
		CCtau /= wsum;
		SStau /= wsum;
		YCtau -= Y * Ctau;
		YStau -= Y * Stau;
		CCtau -= Ctau * Ctau;
		SStau -= Stau * Stau;
		YY -= Y * Y;
		


    	pgram[tid] = (YCtau * YCtau / CCtau + YStau * YStau / SStau) / YY;	    



	    }	
}