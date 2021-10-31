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


#include <math.h>

#define BLOCKSIZE 512 //CUDA block size

//structs
struct lookupObj
{
	unsigned int objId;
	unsigned int idxMin;
	unsigned int idxMax;
};


template <typename DTYPE>
__global__ void lombscarglegpuonethread(DTYPE * x, DTYPE * y, DTYPE * freqs, unsigned int * sizexy, unsigned int * sizefreqs, DTYPE * pgram);

template <typename DTYPE>
__global__ void lombscargleFindMaxPowers(struct lookupObj * objectLookup, DTYPE * pgram, unsigned int * sizefreqs, DTYPE * periodsList, DTYPE * maxPower);

template <typename DTYPE>
__forceinline__ __device__ void parReductionMaximumPowerinSM(DTYPE maxPowerForComputingPeriod[], unsigned int maxPowerIdxForComputingPeriod[]);

template <typename DTYPE>
__global__ void lombscargleBatch(DTYPE * x, DTYPE * y, struct lookupObj * objectLookup, DTYPE * pgram,  DTYPE * foundPeriod, const double minFreq, const double maxFreq, const unsigned int numFreqs);

template <typename DTYPE>
__global__ void lombscargleBatchReturnPgram(DTYPE * x, DTYPE * y, struct lookupObj * objectLookup, DTYPE * pgram,  DTYPE * foundPeriod, const double minFreq, const double maxFreq, const unsigned int numFreqs);

// template <typename DTYPE>
// __global__ void lombscargleBatchSM(DTYPE * x, DTYPE * y, struct lookupObj * objectLookup, DTYPE * pgram,  DTYPE * foundPeriod, const double minFreq, const double maxFreq, const unsigned int numFreqs);

template <typename DTYPE>
__global__ void lombscargleOneObject(DTYPE * x, DTYPE * y, DTYPE * pgram,  const unsigned int sizeData, const double minFreq, const double maxFreq, const unsigned int numFreqs);

template <typename DTYPE>
__global__ void lombscargleOneObjectSM(DTYPE * x, DTYPE * y, DTYPE * pgram,  const unsigned int sizeData, const double minFreq, const double maxFreq, const unsigned int numFreqs);

template <typename DTYPE>
__global__ void lombscargleBatchError(DTYPE * x, DTYPE * y, DTYPE * dy, struct lookupObj * objectLookup, DTYPE * pgram,  DTYPE * foundPeriod, 
	const double minFreq, const double maxFreq, const unsigned int numFreqs);

template <typename DTYPE>
__global__ void lombscargleOneObjectError(DTYPE * x, DTYPE * y, DTYPE * dy, DTYPE * pgram, const unsigned int sizeData, const double minFreq, const double maxFreq, const unsigned int numFreqs);

__forceinline__ __device__ float atan(float a, float b){
	return atan2f(a,b);
}

__forceinline__ __device__ double atan(double a, double b){
	return atan2(a,b);
}

template <typename DTYPE>
__forceinline__ __device__ void parReductionMaximumPowerinSM(DTYPE maxPowerForComputingPeriod[], unsigned int maxPowerIdxForComputingPeriod[])
{		
		int i = blockDim.x / 2;
    	while (i != 0) {
	      	if(threadIdx.x < i && maxPowerForComputingPeriod[threadIdx.x + i] > maxPowerForComputingPeriod[threadIdx.x])
	      	{
	        maxPowerForComputingPeriod[threadIdx.x] = maxPowerForComputingPeriod[threadIdx.x + i]; 
	        maxPowerIdxForComputingPeriod[threadIdx.x] = maxPowerIdxForComputingPeriod[threadIdx.x + i];
	    	}
	    	__syncthreads();
	    	i/=2;
    	}    	
}




//lombsscarge with 1 thread for testing
template <typename DTYPE>
__global__ void lombscarglegpuonethread(DTYPE * x, DTYPE * y, DTYPE * freqs, unsigned int * sizexy, unsigned int * sizefreqs, DTYPE * pgram)
{
	int tid=threadIdx.x+ (blockIdx.x*blockDim.x); 

	if (tid==0){
		
	


    
	    double c, s, xc, xs, cc, ss, cs;
	    double tau, c_tau, s_tau, c_tau2, s_tau2, cs_tau;

		for (int i=0; i<*sizefreqs; i++)
		{

	        xc = 0.0;
	        xs = 0.0;
	        cc = 0.0;
	        ss = 0.0;
	        cs = 0.0;

	        for (int j=0; j<*sizexy; j++)
	        {	
	            c = cos(freqs[i] * x[j]);
	            s = sin(freqs[i] * x[j]);

	            xc += y[j] * c;
	            xs += y[j] * s;
	            cc += c * c;
	            ss += s * s;
	            cs += c * s;
	        }
	            
	        tau = atan(2.0 * cs, cc - ss) / (2.0 * freqs[i]);
	        c_tau = cos(freqs[i] * tau);
	        s_tau = sin(freqs[i] * tau);
	        c_tau2 = c_tau * c_tau;
	        s_tau2 = s_tau * s_tau;
	        cs_tau = 2.0 * c_tau * s_tau;

	        pgram[i] = 0.5 * (((c_tau * xc + s_tau * xs)*(c_tau * xc + s_tau * xs) / \
	            (c_tau2 * cc + cs_tau * cs + s_tau2 * ss)) + \
	            ((c_tau * xs - s_tau * xc)*(c_tau * xs - s_tau * xc) / \
	            (c_tau2 * ss - cs_tau * cs + s_tau2 * cc)));

	    }

    }	
	
}



//Each thread stores the powers of the pgram and global memory and then have another kernel find the maximum index corresponding to the correct power
template <typename DTYPE>
__global__ void lombscargleOneObjectError(DTYPE * x, DTYPE * y, DTYPE * dy, DTYPE * pgram, const unsigned int sizeData, const double minFreq, const double maxFreq, const unsigned int numFreqs)
{

	unsigned int tid=(blockIdx.x*blockDim.x)+threadIdx.x; 	
	
	//Values shared by all threads in the block
	__shared__ DTYPE freqStep; //increment the frequency by this much

	//Set the values in shared memory using thread 0
	if (threadIdx.x==0)
	{
	freqStep=(maxFreq-minFreq)/(numFreqs*1.0);	
	}
	
	__syncthreads();


		DTYPE w, sin_omega_t, cos_omega_t, S, C, S2, C2, tau, Y, wsum, YY, Stau, Ctau, YCtau, YStau, CCtau, SStau; 
		
		wsum = 0.0;
		S = 0.0;
		C = 0.0;
		S2 = 0.0;
		C2 = 0.0;

		if (tid<numFreqs)
		{

		DTYPE freqToTest=minFreq+(freqStep*tid);	
    
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

		tau = 0.5 * atan(S2, C2) / freqToTest;

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



//Each thread stores the powers of the pgram and global memory and then have another kernel find the maximum index corresponding to the correct power
template <typename DTYPE>
__global__ void lombscargleOneObject(DTYPE * x, DTYPE * y, DTYPE * pgram, const unsigned int sizeData, const double minFreq, const double maxFreq, const unsigned int numFreqs)
{

	unsigned int tid=(blockIdx.x*blockDim.x)+threadIdx.x; 	
	
	//Values shared by all threads in the block
	__shared__ DTYPE freqStep; //increment the frequency by this much

	//Set the values in shared memory using thread 0
	if (threadIdx.x==0)
	{
	freqStep=(maxFreq-minFreq)/(numFreqs*1.0);	
	}
	
	__syncthreads();


    	DTYPE c, s;
	    DTYPE tau, c_tau, s_tau, c_tau2, s_tau2, cs_tau;		
		
			if (tid<numFreqs)
			{


			DTYPE freqToTest=minFreq+(freqStep*tid);

	        DTYPE xc = 0.0;
	        DTYPE xs = 0.0;
	        DTYPE cc = 0.0;
	        DTYPE ss = 0.0;
	        DTYPE cs = 0.0;

	
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
	        
	        tau = atan(2.0 * cs, cc - ss) / (2.0 * freqToTest);

	        sincos(freqToTest * tau, &s_tau, &c_tau);	       

	        c_tau2 = c_tau * c_tau;
	        s_tau2 = s_tau * s_tau;
	        cs_tau = 2.0 * c_tau * s_tau;

	        

	        // DTYPE power=0.5 * (((c_tau * xc + s_tau * xs)*(c_tau * xc + s_tau * xs) / \
	        //     (c_tau2 * cc + cs_tau * cs + s_tau2 * ss)) + \
	        //     ((c_tau * xs - s_tau * xc)*(c_tau * xs - s_tau * xc) / \
	        //     (c_tau2 * ss - cs_tau * cs + s_tau2 * cc)));
	        
	        DTYPE f1=(c_tau * xc + s_tau * xs);
	        DTYPE f2=(c_tau * xs - s_tau * xc);
	        DTYPE d1=(f1*f1);
	        DTYPE d2=(c_tau2 * cc + cs_tau * cs + s_tau2 * ss);
	        DTYPE d4=(f2*f2);
	        DTYPE d5=(c_tau2 * ss - cs_tau * cs + s_tau2 * cc);

	        pgram[tid]=0.5 * ((d1/d2)+(d4/d5));

	    	}	
}








//Each thread stores the powers of the pgram and global memory and then have another kernel find the maximum index corresponding to the correct power
template <typename DTYPE>
__global__ void lombscargleOneObjectSM(DTYPE * x, DTYPE * y, DTYPE * pgram, const unsigned int sizeData, const double minFreq, const double maxFreq, const unsigned int numFreqs)
{

	unsigned int tid=(blockIdx.x*blockDim.x)+threadIdx.x; 	
	
	//Values shared by all threads in the block
	__shared__ DTYPE freqStep; //increment the frequency by this much

	//SM for reading the x and y values
	__shared__ DTYPE x_SM[BLOCKSIZE];
	__shared__ DTYPE y_SM[BLOCKSIZE];

	//Set the values in shared memory using thread 0
	if (threadIdx.x==0)
	{
	freqStep=(maxFreq-minFreq)/(numFreqs*1.0);	
	}
	
	__syncthreads();


    	DTYPE c, s;
	    DTYPE tau, c_tau, s_tau, c_tau2, s_tau2, cs_tau;
	

			DTYPE freqToTest=minFreq+(freqStep*tid);

	        DTYPE xc = 0.0;
	        DTYPE xs = 0.0;
	        DTYPE cc = 0.0;
	        DTYPE ss = 0.0;
	        DTYPE cs = 0.0;

	

	        for (int j=0; j<sizeData; j+=blockDim.x)
	        {	

	        	__syncthreads();
	        	//read x and y values into SM
	        	if ((j+threadIdx.x)<=sizeData)
	        	{
	        	x_SM[threadIdx.x]=x[j+threadIdx.x];
	        	y_SM[threadIdx.x]=y[j+threadIdx.x];	
	        	}

	        	__syncthreads();

	        	for (int k=0; k<BLOCKSIZE && ((j+k)<=sizeData); k++)
	         	{
	        	
	            sincos(freqToTest * x_SM[k], &s, &c);


	            xc += y_SM[k] * c;
	            xs += y_SM[k] * s;
	            cc += c * c;
	            ss += s * s;
	            cs += c * s;
	        	}
	        	__syncthreads();
	        }
	        
	        if (tid<numFreqs)
			{
	        tau = atan(2.0 * cs, cc - ss) / (2.0 * freqToTest);


	        sincos(freqToTest * tau, &s_tau, &c_tau);	       

	        c_tau2 = c_tau * c_tau;
	        s_tau2 = s_tau * s_tau;
	        cs_tau = 2.0 * c_tau * s_tau;

	        

	        // DTYPE power=0.5 * (((c_tau * xc + s_tau * xs)*(c_tau * xc + s_tau * xs) / \
	        //     (c_tau2 * cc + cs_tau * cs + s_tau2 * ss)) + \
	        //     ((c_tau * xs - s_tau * xc)*(c_tau * xs - s_tau * xc) / \
	        //     (c_tau2 * ss - cs_tau * cs + s_tau2 * cc)));
	        
	        DTYPE f1=(c_tau * xc + s_tau * xs);
	        DTYPE f2=(c_tau * xs - s_tau * xc);
	        DTYPE d1=(f1*f1);
	        DTYPE d2=(c_tau2 * cc + cs_tau * cs + s_tau2 * ss);
	        DTYPE d4=(f2*f2);
	        DTYPE d5=(c_tau2 * ss - cs_tau * cs + s_tau2 * cc);

	        pgram[tid]=0.5 * ((d1/d2)+(d4/d5));

	    	}	
}





//The kernel with error only uses global memory and returns the pgram
template <typename DTYPE>
__global__ void lombscargleBatchError(DTYPE * x, DTYPE * y, DTYPE * dy, struct lookupObj * objectLookup, DTYPE * pgram,  DTYPE * foundPeriod, 
	const double minFreq, const double maxFreq, const unsigned int numFreqs)
{
	
	
	
	//Values shared by all threads in the block
	__shared__ DTYPE freqStep; //increment the frequency by this much
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


	DTYPE w, sin_omega_t, cos_omega_t, S, C, S2, C2, tau, Y, wsum, YY, Stau, Ctau, YCtau, YStau, CCtau, SStau; 



	for (int i=0; i<numFreqs && ((i+threadIdx.x)<numFreqs); i+=blockDim.x)
	{


	int idx=i+threadIdx.x;
	DTYPE freqToTest=minFreq+(freqStep*idx);

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

        tau = 0.5 * atan(S2, C2) / freqToTest;

		
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




//Compute frequency on the fly in the kernel gievn frequency ranges
template <typename DTYPE>
__global__ void lombscargleBatchReturnPgram(DTYPE * x, DTYPE * y, struct lookupObj * objectLookup, DTYPE * pgram,  DTYPE * foundPeriod, 
	const double minFreq, const double maxFreq, const unsigned int numFreqs)
{
	
	
	
	//Values shared by all threads in the block
	__shared__ DTYPE freqStep; //increment the frequency by this much

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


    	DTYPE c, s;
	    DTYPE tau, c_tau, s_tau, c_tau2, s_tau2, cs_tau;
	

		for (int i=0; i<numFreqs && ((i+threadIdx.x)<numFreqs); i+=blockDim.x)
		{
			int idx=i+threadIdx.x;


			DTYPE freqToTest=minFreq+(freqStep*idx);

	        DTYPE xc = 0.0;
	        DTYPE xs = 0.0;
	        DTYPE cc = 0.0;
	        DTYPE ss = 0.0;
	        DTYPE cs = 0.0;

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
	        

	        tau = atan(2.0 * cs, cc - ss) / (2.0 * freqToTest);
	        
	        sincos(freqToTest * tau, &s_tau, &c_tau);	       
	        
	        

	        c_tau2 = c_tau * c_tau;
	        s_tau2 = s_tau * s_tau;
	        cs_tau = 2.0 * c_tau * s_tau;

	        

	        // DTYPE power=0.5 * (((c_tau * xc + s_tau * xs)*(c_tau * xc + s_tau * xs) / \
	        //     (c_tau2 * cc + cs_tau * cs + s_tau2 * ss)) + \
	        //     ((c_tau * xs - s_tau * xc)*(c_tau * xs - s_tau * xc) / \
	        //     (c_tau2 * ss - cs_tau * cs + s_tau2 * cc)));
	        
	        DTYPE f1=(c_tau * xc + s_tau * xs);
	        DTYPE f2=(c_tau * xs - s_tau * xc);
	        DTYPE d1=(f1*f1);
	        DTYPE d2=(c_tau2 * cc + cs_tau * cs + s_tau2 * ss);
	        DTYPE d4=(f2*f2);
	        DTYPE d5=(c_tau2 * ss - cs_tau * cs + s_tau2 * cc);

	        DTYPE power=0.5 * ((d1/d2)+(d4/d5));



	        //this is if we want to store the pgram

	        pgram[idx+periodWriteOffset] = power;

	    }
	
}



//Compute frequency on the fly in the kernel gievn frequency ranges
template <typename DTYPE>
__global__ void lombscargleBatch(DTYPE * x, DTYPE * y, struct lookupObj * objectLookup, DTYPE * pgram,  DTYPE * foundPeriod, 
	const double minFreq, const double maxFreq, const unsigned int numFreqs)
{
	
	
	
	//Values shared by all threads in the block
	__shared__ DTYPE freqStep; //increment the frequency by this much

	__shared__ unsigned int minDataIdx;
	__shared__ unsigned int maxDataIdx;
	
	

	//one per thread in the block
	__shared__ DTYPE localMaxPowerFound[BLOCKSIZE];
	__shared__ unsigned int localMaxPowerIdx[BLOCKSIZE];		
	__shared__ DTYPE maxPowerForComputingPeriod[BLOCKSIZE];
	__shared__ unsigned int maxPowerIdxForComputingPeriod[BLOCKSIZE];
	//initialize shared memory arrays
	localMaxPowerFound[threadIdx.x]=0;
	localMaxPowerIdx[threadIdx.x]=0;
	maxPowerForComputingPeriod[threadIdx.x]=0;
	maxPowerIdxForComputingPeriod[threadIdx.x]=0;
	

	//Set the values in shared memory using thread 0
	if (threadIdx.x==0)
	{
	freqStep=(maxFreq-minFreq)/(numFreqs*1.0);	
	minDataIdx=objectLookup[blockIdx.x].idxMin;	
	maxDataIdx=objectLookup[blockIdx.x].idxMax;
	
	}
	
	__syncthreads();


    	DTYPE c, s;
	    DTYPE tau, c_tau, s_tau, c_tau2, s_tau2, cs_tau;
	

		for (int i=0; i<numFreqs && ((i+threadIdx.x)<numFreqs); i+=blockDim.x)
		{
			int idx=i+threadIdx.x;


			DTYPE freqToTest=minFreq+(freqStep*idx);

	        DTYPE xc = 0.0;
	        DTYPE xs = 0.0;
	        DTYPE cc = 0.0;
	        DTYPE ss = 0.0;
	        DTYPE cs = 0.0;

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
	        

	        tau = atan(2.0 * cs, cc - ss) / (2.0 * freqToTest);
	        
	        sincos(freqToTest * tau, &s_tau, &c_tau);	       
	        
	        

	        c_tau2 = c_tau * c_tau;
	        s_tau2 = s_tau * s_tau;
	        cs_tau = 2.0 * c_tau * s_tau;

	        

	        // DTYPE power=0.5 * (((c_tau * xc + s_tau * xs)*(c_tau * xc + s_tau * xs) / \
	        //     (c_tau2 * cc + cs_tau * cs + s_tau2 * ss)) + \
	        //     ((c_tau * xs - s_tau * xc)*(c_tau * xs - s_tau * xc) / \
	        //     (c_tau2 * ss - cs_tau * cs + s_tau2 * cc)));
	        
	        DTYPE f1=(c_tau * xc + s_tau * xs);
	        DTYPE f2=(c_tau * xs - s_tau * xc);
	        DTYPE d1=(f1*f1);
	        DTYPE d2=(c_tau2 * cc + cs_tau * cs + s_tau2 * ss);
	        DTYPE d4=(f2*f2);
	        DTYPE d5=(c_tau2 * ss - cs_tau * cs + s_tau2 * cc);

	        DTYPE power=0.5 * ((d1/d2)+(d4/d5));




	        //this is if we don't want to store the pgram 
	        if (localMaxPowerFound[threadIdx.x]<power)
	        {
	        	localMaxPowerIdx[threadIdx.x]=idx;
	        	localMaxPowerFound[threadIdx.x]=power;
	        }


	    }


	    //when not using the pgram
	    //all threads store their maximum power found in SM
	    //This will only work when there's one block
	    maxPowerForComputingPeriod[threadIdx.x]=localMaxPowerFound[threadIdx.x];
	    maxPowerIdxForComputingPeriod[threadIdx.x]=localMaxPowerIdx[threadIdx.x];

	    __syncthreads();
	    //Parallel reduction to find the maximum power
	 	parReductionMaximumPowerinSM(maxPowerForComputingPeriod, maxPowerIdxForComputingPeriod);
    	__syncthreads();
    	if (threadIdx.x==0)
    	{
    	foundPeriod[blockIdx.x]=1.0/(minFreq+(maxPowerIdxForComputingPeriod[threadIdx.x]*(freqStep*1.0)))*2*M_PI;
    	}

	
}


//Compute frequency on the fly in the kernel gievn frequency ranges
//Use SM to store the data
// template <typename DTYPE>
// __global__ void lombscargleBatchSM(DTYPE * x, DTYPE * y, struct lookupObj * objectLookup, DTYPE * pgram,  DTYPE * foundPeriod, 
// 	const double minFreq, const double maxFreq, const unsigned int numFreqs)
// {
	
	
	
// 	//Values shared by all threads in the block
// 	__shared__ DTYPE freqStep; //increment the frequency by this much

// 	__shared__ unsigned int minDataIdx;
// 	__shared__ unsigned int maxDataIdx;

// 	//testing SM for reading the x and y values
// 	__shared__ DTYPE x_SM[BLOCKSIZE];
// 	__shared__ DTYPE y_SM[BLOCKSIZE];
	
// 	#if RETURNPGRAM==1
// 	__shared__ unsigned int periodWriteOffset;
// 	#endif

// 	//one per thread in the block
// 	#if RETURNPGRAM==0
// 	__shared__ DTYPE localMaxPowerFound[BLOCKSIZE];
// 	__shared__ unsigned int localMaxPowerIdx[BLOCKSIZE];		
// 	__shared__ DTYPE maxPowerForComputingPeriod[BLOCKSIZE];
// 	__shared__ unsigned int maxPowerIdxForComputingPeriod[BLOCKSIZE];
// 	//initialize shared memory arrays
// 	localMaxPowerFound[threadIdx.x]=0;
// 	localMaxPowerIdx[threadIdx.x]=0;
// 	maxPowerForComputingPeriod[threadIdx.x]=0;
// 	maxPowerIdxForComputingPeriod[threadIdx.x]=0;
// 	#endif
	

// 	//Set the values in shared memory using thread 0
// 	if (threadIdx.x==0)
// 	{
// 	freqStep=(maxFreq-minFreq)/(numFreqs*1.0);	
// 	minDataIdx=objectLookup[blockIdx.x].idxMin;	
// 	maxDataIdx=objectLookup[blockIdx.x].idxMax;
// 	#if RETURNPGRAM==1
// 	periodWriteOffset=blockIdx.x*numFreqs;
// 	#endif
	
// 	}
	
// 	__syncthreads();


//     	DTYPE c, s;
// 	    DTYPE tau, c_tau, s_tau, c_tau2, s_tau2, cs_tau;
	

// 	    // for (int i=0; i<numFreqs && ((i+threadIdx.x)<numFreqs); i+=blockDim.x)
// 		for (int i=0; i<numFreqs; i+=blockDim.x)
// 		{

			
// 			int idx=i+threadIdx.x;


// 			DTYPE freqToTest=minFreq+(freqStep*idx);

// 	        DTYPE xc = 0.0;
// 	        DTYPE xs = 0.0;
// 	        DTYPE cc = 0.0;
// 	        DTYPE ss = 0.0;
// 	        DTYPE cs = 0.0;
	    	
	        
	        
// 	        for (int j=minDataIdx; j<=maxDataIdx; j+=BLOCKSIZE)
// 	        {
// 	        	__syncthreads();
// 	        	//read x and y values into SM
// 	        	if ((j+threadIdx.x)<=maxDataIdx)
// 	        	{
// 	        	x_SM[threadIdx.x]=x[j+threadIdx.x];
// 	        	y_SM[threadIdx.x]=y[j+threadIdx.x];	
// 	        	}
	         
// 	         	__syncthreads();
// 	         	for (int k=0; k<BLOCKSIZE && ((j+k)<=maxDataIdx); k++)
// 	         	{
	        	
// 	            sincos(freqToTest * x_SM[k], &s, &c);


// 	            xc += y_SM[k] * c;
// 	            xs += y_SM[k] * s;
// 	            cc += c * c;
// 	            ss += s * s;
// 	            cs += c * s;
// 	        	}
// 	        	__syncthreads();
// 	        }
	        
// 	        //SM guard
// 	        if ( (i+threadIdx.x)<numFreqs)
// 			{

// 	        tau = atan(2.0 * cs, cc - ss) / (2.0 * freqToTest);

	        
// 	        sincos(freqToTest * tau, &s_tau, &c_tau);	       
	        
	        

// 	        c_tau2 = c_tau * c_tau;
// 	        s_tau2 = s_tau * s_tau;
// 	        cs_tau = 2.0 * c_tau * s_tau;

	        

// 	        // DTYPE power=0.5 * (((c_tau * xc + s_tau * xs)*(c_tau * xc + s_tau * xs) / \
// 	        //     (c_tau2 * cc + cs_tau * cs + s_tau2 * ss)) + \
// 	        //     ((c_tau * xs - s_tau * xc)*(c_tau * xs - s_tau * xc) / \
// 	        //     (c_tau2 * ss - cs_tau * cs + s_tau2 * cc)));
	        
// 	        DTYPE f1=(c_tau * xc + s_tau * xs);
// 	        DTYPE f2=(c_tau * xs - s_tau * xc);
// 	        DTYPE d1=(f1*f1);
// 	        DTYPE d2=(c_tau2 * cc + cs_tau * cs + s_tau2 * ss);
// 	        DTYPE d4=(f2*f2);
// 	        DTYPE d5=(c_tau2 * ss - cs_tau * cs + s_tau2 * cc);

// 	        DTYPE power=0.5 * ((d1/d2)+(d4/d5));



// 	        //this is if we want to store the pgram
// 	        #if RETURNPGRAM==1
// 	        pgram[idx+periodWriteOffset] = power;
// 	        #endif

// 	        //this is if we don't want to store the pgram 
// 	        #if RETURNPGRAM==0
// 	        if (localMaxPowerFound[threadIdx.x]<power)
// 	        {
// 	        	localMaxPowerIdx[threadIdx.x]=idx;
// 	        	localMaxPowerFound[threadIdx.x]=power;
// 	        }
// 	        #endif

// 	    }


// 	    #if RETURNPGRAM==0
// 	    //when not using the pgram
// 	    //all threads store their maximum power found in SM
// 	    //This will only work when there's one block
// 	    maxPowerForComputingPeriod[threadIdx.x]=localMaxPowerFound[threadIdx.x];
// 	    maxPowerIdxForComputingPeriod[threadIdx.x]=localMaxPowerIdx[threadIdx.x];

// 	    __syncthreads();
// 	    //Parallel reduction to find the maximum power
// 	 	parReductionMaximumPowerinSM(maxPowerForComputingPeriod, maxPowerIdxForComputingPeriod);
//     	__syncthreads();
//     	if (threadIdx.x==0)
//     	{
//     	foundPeriod[blockIdx.x]=1.0/(minFreq+(maxPowerIdxForComputingPeriod[threadIdx.x]*(freqStep*1.0)))*2*M_PI;
//     	}
	
// 		#endif


//     	} 
	
// }







