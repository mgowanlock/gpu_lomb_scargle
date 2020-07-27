#include "kernel.h"





//Each thread stores the powers of the pgram and global memory and then have another kernel find the maximum index corresponding to the correct power
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
	        
	        #if DTYPE==float
	        tau = atan2f(2.0 * cs, cc - ss) / (2.0 * freqToTest);
	        #elif DTYPE==double
	        tau = atan2(2.0 * cs, cc - ss) / (2.0 * freqToTest);
	        #endif

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















//Compute frequency on the fly in the kernel gievn frequency ranges
__global__ void lombscargleBatch(DTYPE * x, DTYPE * y, struct lookupObj * objectLookup, DTYPE * pgram,  DTYPE * foundPeriod, 
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
	        
	        #if DTYPE==float
	        tau = atan2f(2.0 * cs, cc - ss) / (2.0 * freqToTest);
	        #elif DTYPE==double
	        tau = atan2(2.0 * cs, cc - ss) / (2.0 * freqToTest);
	        #endif

	        
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
	        pgram[idx+periodWriteOffset] = power;
	    }
	
}





