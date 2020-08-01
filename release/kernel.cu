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








//The kernel with error only uses global memory and returns the pgram
//generalized L-S from AstroPy
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
		#if DTYPE==float
        tau = 0.5 * atan2f(S2, C2) / freqToTest;
        #elif DTYPE==double
        tau = 0.5 * atan2(S2, C2) / freqToTest;
        #endif
		
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
		#if DTYPE==float
		tau = 0.5 * atan2f(S2, C2) / freqToTest;
		#elif DTYPE==double
		tau = 0.5 * atan2(S2, C2) / freqToTest;
		#endif
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