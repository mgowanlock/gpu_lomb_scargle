#include <fstream>
#include <istream>
#include <iostream>
#include <string>
#include <string.h>
#include <sstream>
#include <cstdlib>
#include <stdio.h>
#include <random>
#include "omp.h"
#include <algorithm> 
#include <queue>
#include <iomanip>
#include <set>
#include <algorithm>
#include <thrust/sort.h>
#include <thrust/host_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/device_vector.h>
#include <cuda_profiler_api.h>
#include <thrust/extrema.h>
#include "structs.h"
#include "params.h"

#include "kernel.h"


//for printing defines as strings

#define STR_HELPER(x) #x
#define STR(x) STR_HELPER(x)

//Error checking GPU calls
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}


//function prototypes
void importObjXYData(char * fnamedata, unsigned int * sizeData, unsigned int ** objectId, DTYPE ** timeX, DTYPE ** magY, DTYPE ** magDY);

//CPU L-S Functions:
void lombscarglecpu(bool mode, DTYPE * x, DTYPE * y, const unsigned int sizeData, const unsigned int numFreqs, const DTYPE minFreq, const DTYPE maxFreq, const DTYPE freqStep, DTYPE * pgram);
void lombscarglecpuinnerloop(int iteration, DTYPE * x, DTYPE * y, DTYPE * pgram, DTYPE * freqToTest, const unsigned int sizeData);
void lombscargleCPUOneObject(DTYPE * timeX,  DTYPE * magY, unsigned int * sizeData, const DTYPE minFreq, const DTYPE maxFreq, const unsigned int numFreqs, DTYPE * foundPeriod, DTYPE * pgram);
void lombscargleCPUBatch(unsigned int * objectId, DTYPE * timeX,  DTYPE * magY, unsigned int * sizeData, const DTYPE minFreq, const DTYPE maxFreq, const unsigned int numFreqs, DTYPE * sumPeriods, DTYPE * pgram, DTYPE * foundPeriod);

//With error
void lombscargleCPUOneObjectError(DTYPE * time, DTYPE * magY, DTYPE * magDY, unsigned int * sizeData, const DTYPE minFreq, const DTYPE maxFreq, const unsigned int numFreqs, DTYPE * foundPeriod, DTYPE * pgram);
void lombscarglecpuError(bool mode, DTYPE * x, DTYPE * y, DTYPE *dy, const unsigned int sizeData, const unsigned int numFreqs, const DTYPE minFreq, const DTYPE maxFreq, const DTYPE freqStep, DTYPE * pgram);
void lombscarglecpuinnerloopAstroPy(int iteration, DTYPE * x, DTYPE * y, DTYPE * dy, DTYPE * pgram, DTYPE * freqToTest, const unsigned int sizeData);
void lombscargleCPUBatchError(unsigned int * objectId, DTYPE * timeX,  DTYPE * magY, DTYPE * magDY, unsigned int * sizeData, const DTYPE minFreq, const DTYPE maxFreq, const unsigned int numFreqs, DTYPE * sumPeriods, DTYPE * pgram, DTYPE * foundPeriod);
void updateYerrorfactor(DTYPE * y, DTYPE *dy, const unsigned int sizeData);

//GPU functions
void batchGPULS(unsigned int * objectId, DTYPE * timeX,  DTYPE * magY, DTYPE * magDY, unsigned int * sizeData, const DTYPE minFreq, const DTYPE maxFreq, const unsigned int numFreqs, DTYPE * sumPeriods, DTYPE ** pgram, DTYPE * foundPeriod);
void GPULSOneObject(DTYPE * timeX,  DTYPE * magY, DTYPE * magDY, unsigned int * sizeData, const DTYPE minFreq, const DTYPE maxFreq, const unsigned int numFreqs, DTYPE * periodFound, DTYPE ** pgram);
void computeObjectRanges(unsigned int * objectId, unsigned int * sizeData, struct lookupObj ** objectLookup, unsigned int * numUniqueObjects);
void pinnedMemoryCopyDtoH(DTYPE * pinned_buffer, unsigned int sizeBufferElems, DTYPE * dev_data, DTYPE * pageable, unsigned int sizeTotalData);

void computePeriod(DTYPE * pgram, const unsigned int numFreqs, const DTYPE minFreq, const DTYPE freqStep, DTYPE * foundPeriod);

void warmUpGPU();


using namespace std;

int main(int argc, char *argv[])
{

	warmUpGPU();
	cudaProfilerStart();
	omp_set_nested(1);

	//validation and output to file
	char fname[]="gpu_stats.txt";
	ofstream gpu_stats;
	gpu_stats.open(fname,ios::app);	

	/////////////////////////
	// Get information from command line
	/////////////////////////

	
	//Input data filename (objId, time, amplitude), minimum frequency, maximum frequency, numer of frequencies, mode
	if (argc!=6)
	{
	cout <<"\n\nIncorrect number of input parameters.\nExpected values: Data filename (objId, time, amplitude), minimum frequency, maximum frequency, number of frequencies, mode\n";
	return 0;
	}
	
	
	char inputFname[500];
	strcpy(inputFname,argv[1]);
	double minFreq=atof(argv[2]); //inclusive
	double maxFreq=atof(argv[3]); //exclusive
	const unsigned int freqToTest=atoi(argv[4]);
    int MODE = atoi(argv[5]);

	printf("\nDataset file: %s",inputFname);
	printf("\nMinimum Frequency: %f",minFreq);
	printf("\nMaximum Frequency: %f",maxFreq);
	printf("\nNumber of frequencies to test: %u", freqToTest);
	printf("\nMode: %d", MODE);

	#if ERROR==1
	printf("\nExecuting L-S variant from AstroPy that propogates error and floats the mean");
	#endif
	
	
	/////////////
	//Import Data
	/////////////
	unsigned int * objectId=NULL; 
	DTYPE * timeX=NULL; 
	DTYPE * magY=NULL;
	DTYPE * magDY=NULL;
	unsigned int sizeData;
	importObjXYData(inputFname, &sizeData, &objectId, &timeX, &magY, &magDY);	
	
	//pgram allocated in the functions below
	//Stores the LS power for each frequency
	DTYPE * pgram=NULL;

	//foundPeriod is allocated in the batch functions below
	DTYPE * foundPeriod=NULL;

	//Batch of LS to compute on the GPU
	if (MODE==1)
	{
		DTYPE sumPeriods=0;
		
		double tstart=omp_get_wtime();
		
		batchGPULS(objectId, timeX, magY, magDY, &sizeData, minFreq, maxFreq, freqToTest, &sumPeriods, &pgram, foundPeriod);
		
		double tend=omp_get_wtime();
		double totalTime=tend-tstart;
		printf("\nTotal time to compute batch: %f", totalTime);
		printf("\n[Validation] Sum of all periods: %f", sumPeriods);

		gpu_stats<<totalTime<<", "<< inputFname<<", Sum of periods: "<<sumPeriods<<", Min/Max Freq: "<<minFreq<<"/"<<maxFreq<<",  Num tested freq: "<<freqToTest<<", MODE: "<<MODE<<", NTHREADSCPU/BLOCKSIZE/ERROR/SHMEM/RETURNPGRAM/PINNED/SIZEPINNEDBUFFERMIB/NSTREAMS/DTYPE: "<<NTHREADSCPU<<", "<<BLOCKSIZE<<", "<<ERROR<<", "<<SHMEM<<", "<<RETURNPGRAM<<", "<<PINNED<<", "<<SIZEPINNEDBUFFERMIB<<", "<<NSTREAMS<<", "<<STR(DTYPE)<<endl;
	}
	//One object to compute on the GPU
	else if (MODE==2)
	{
		DTYPE periodFound=0;	
		double tstart=omp_get_wtime();
		
		#if ERROR==1
		updateYerrorfactor(magY, magDY, sizeData);
		#endif	

		GPULSOneObject(timeX, magY, magDY, &sizeData, minFreq, maxFreq, freqToTest, &periodFound, &pgram);
		
		double tend=omp_get_wtime();
		double totalTime=tend-tstart;
		printf("\nTotal time to compute batch: %f", totalTime);
		printf("\n[Validation] Period: %f", periodFound);

		gpu_stats<<totalTime<<", "<< inputFname<<", Sum of periods: "<<periodFound<<", Min/Max Freq: "<<minFreq<<"/"<<maxFreq<<",  Num tested freq: "<<freqToTest<<", MODE: "<<MODE<<", NTHREADSCPU/BLOCKSIZE/ERROR/SHMEM/RETURNPGRAM/PINNED/SIZEPINNEDBUFFERMIB/NSTREAMS/DTYPE: "<<NTHREADSCPU<<", "<<BLOCKSIZE<<", "<<ERROR<<", "<<SHMEM<<", "<<RETURNPGRAM<<", "<<PINNED<<", "<<SIZEPINNEDBUFFERMIB<<", "<<NSTREAMS<<", "<<STR(DTYPE)<<endl;
	}
	//CPU- batch processing
	else if (MODE==4)
	{
		DTYPE sumPeriods=0;
		double tstart=omp_get_wtime();
		#if ERROR==0
		lombscargleCPUBatch(objectId, timeX, magY, &sizeData, minFreq, maxFreq, freqToTest, &sumPeriods, pgram, foundPeriod);
		#endif
		#if ERROR==1
		lombscargleCPUBatchError(objectId, timeX, magY, magDY, &sizeData, minFreq, maxFreq, freqToTest, &sumPeriods, pgram, foundPeriod);
		#endif
		double tend=omp_get_wtime();
		double totalTime=tend-tstart;
		printf("\nTotal time to compute batch: %f", totalTime);
		printf("\n[Validation] Sum of all periods: %f", sumPeriods);
		gpu_stats<<totalTime<<", "<< inputFname<<", Sum of periods: "<<sumPeriods<<", Min/Max Freq: "<<minFreq<<"/"<<maxFreq<<",  Num tested freq: "<<freqToTest<<", MODE: "<<MODE<<", NTHREADSCPU: "<<NTHREADSCPU<<", ERROR: "<<ERROR<<", DTYPE: "<<STR(DTYPE)<<endl;
	}
	//CPU- one object
	else if (MODE==5)
	{
		DTYPE foundPeriod=0;
		double tstart=omp_get_wtime();
		#if ERROR==0
		lombscargleCPUOneObject(timeX, magY, &sizeData, minFreq, maxFreq, freqToTest, &foundPeriod, pgram);
		#endif
		#if ERROR==1
		lombscargleCPUOneObjectError(timeX, magY, magDY, &sizeData, minFreq, maxFreq, freqToTest, &foundPeriod, pgram);
		#endif
		double tend=omp_get_wtime();
		double totalTime=tend-tstart;
		printf("\nTotal time to compute pgram (one object): %f", totalTime);
		printf("\n[Validation] Period: %f", foundPeriod);
		gpu_stats<<totalTime<<", "<< inputFname<<", Sum of periods: "<<foundPeriod<<", Min/Max Freq: "<<minFreq<<"/"<<maxFreq<<",  Num tested freq: "<<freqToTest<<", MODE: "<<MODE<<", NTHREADSCPU: "<<NTHREADSCPU<<", ERROR: "<<ERROR<<", DTYPE: "<<STR(DTYPE)<<endl;
	}



	//free memory
	free(foundPeriod);
	free(objectId);
	free(timeX);
	free(magY);
	free(magDY);
	free(pgram);


	cudaProfilerStop();

	gpu_stats.close();
	printf("\n");
	return 0;
}


void computeObjectRanges(unsigned int * objectId, unsigned int * sizeData, struct lookupObj ** objectLookup, unsigned int * numUniqueObjects)
{
	//Scan to find unique object ids;
	unsigned int lastId=objectId[0];
	unsigned int cntUnique=1;
	for (unsigned int i=1; i<*sizeData; i++)
	{
		if (lastId!=objectId[i])
		{
			cntUnique++;
			lastId=objectId[i];
		}
	}

	//allocate memory for the struct
	*objectLookup=(lookupObj*)malloc(sizeof(lookupObj)*cntUnique);

	*numUniqueObjects=cntUnique;
	printf("\nUnique objects in file: %u",*numUniqueObjects);



	lastId=objectId[0];
	unsigned int cnt=0;
	for (unsigned int i=1; i<*sizeData; i++)
	{
		if (lastId!=objectId[i])
		{
			(*objectLookup)[cnt].objId=lastId;
			(*objectLookup)[cnt+1].idxMin=i;
			(*objectLookup)[cnt].idxMax=i-1;
			cnt++;
			lastId=objectId[i];
		}
	}

	//first and last ones
	(*objectLookup)[0].idxMin=0;
	(*objectLookup)[cnt].objId=objectId[(*sizeData)-1];
	(*objectLookup)[cnt].idxMax=(*sizeData)-1;

}


void pinnedMemoryCopyDtoH(DTYPE * pinned_buffer, unsigned int sizeBufferElems, DTYPE * dev_data, DTYPE * pageable, unsigned int sizeTotalData)
{

  	cudaStream_t streams[NSTREAMS];
  	//create stream for the device
	for (int i=0; i<NSTREAMS; i++)
	{
	cudaStreamCreate(&streams[i]);
	}
  	

  	unsigned int numIters=sizeTotalData/sizeBufferElems;

  	unsigned int totalelemstransfered=0;
  	#pragma omp parallel for num_threads(NSTREAMS) reduction(+:totalelemstransfered)
  	for (unsigned int i=0; i<=numIters; i++)
  	{
  		int tid=omp_get_thread_num();
	  	unsigned int offsetstart=i*sizeBufferElems;
	  	unsigned int offsetend=(i+1)*sizeBufferElems;
	  	unsigned int elemsToTransfer=sizeBufferElems;
	  	if (offsetend>=sizeTotalData)
	  	{
	  		elemsToTransfer=sizeTotalData-offsetstart; 
	  	}	
	  	totalelemstransfered+=elemsToTransfer;
		
		unsigned int pinnedBufferOffset=tid*sizeBufferElems;
		gpuErrchk(cudaMemcpyAsync(pinned_buffer+pinnedBufferOffset, dev_data+(offsetstart), sizeof(DTYPE)*elemsToTransfer, cudaMemcpyDeviceToHost, streams[tid])); 

		cudaStreamSynchronize(streams[tid]);
		
		//Copy from pinned to pageable memory
		//Nested parallelization with openmp
		#pragma omp parallel for num_threads(4)
		for (unsigned int j=0; j<elemsToTransfer; j++)
		{
			pageable[offsetstart+j]=pinned_buffer[pinnedBufferOffset+j];
		} 	
	
	}

	for (int i=0; i<NSTREAMS; i++)
	{
	cudaStreamDestroy(streams[i]);
	}
}









//Compute pgram for one object, not a batch of objects
void GPULSOneObject(DTYPE * timeX,  DTYPE * magY, DTYPE * magDY, unsigned int * sizeData, const DTYPE minFreq, const DTYPE maxFreq, const unsigned int numFreqs, DTYPE * periodFound, DTYPE ** pgram)
{
	

	DTYPE * dev_timeX;
	DTYPE * dev_magY;
	unsigned int * dev_sizeData;
	DTYPE * dev_pgram;
	DTYPE * dev_foundPeriod;
	
	

	//allocate memory on the GPU
	gpuErrchk(cudaMalloc((void**)&dev_timeX, sizeof(DTYPE)*(*sizeData)));
	gpuErrchk(cudaMalloc((void**)&dev_magY, sizeof(DTYPE)*(*sizeData)));
	
	//If astropy implementation with error
	#if ERROR==1
	DTYPE * dev_magDY;
	gpuErrchk(cudaMalloc((void**)&dev_magDY, sizeof(DTYPE)*(*sizeData)));	
	#endif
	
	// Result periodogram
	//need to allocate it on the GPUeven if we do not return it to the host so that we can find the maximum power
	gpuErrchk(cudaMalloc((void**)&dev_pgram, sizeof(DTYPE)*numFreqs));
	
	

	//the maximum power in the periodogram. Use this when we don't want to return the periodogram
	gpuErrchk(cudaMalloc((void**)&dev_foundPeriod, sizeof(DTYPE)));
	gpuErrchk(cudaMalloc((void**)&dev_sizeData, sizeof(unsigned int)));

	//copy data to the GPU
	gpuErrchk(cudaMemcpy( dev_timeX, timeX, sizeof(DTYPE)*(*sizeData), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy( dev_magY, magY, sizeof(DTYPE)*(*sizeData), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy( dev_sizeData, sizeData, sizeof(unsigned int), cudaMemcpyHostToDevice));

	#if ERROR==1
	gpuErrchk(cudaMemcpy( dev_magDY, magDY, sizeof(DTYPE)*(*sizeData), cudaMemcpyHostToDevice));
	#endif

	const unsigned int szData=*sizeData;
	const unsigned int numBlocks=ceil(numFreqs*1.0/BLOCKSIZE*1.0);
	
	double tstart=omp_get_wtime();
  	//Do lomb-scargle
  	#if ERROR==0 && SHMEM==0
  	lombscargleOneObject<<< numBlocks, BLOCKSIZE>>>(dev_timeX, dev_magY, dev_pgram, szData, minFreq, maxFreq, numFreqs);
  	#elif ERROR==0 && SHMEM==1
  	lombscargleOneObjectSM<<< numBlocks, BLOCKSIZE>>>(dev_timeX, dev_magY, dev_pgram, szData, minFreq, maxFreq, numFreqs);
  	#elif ERROR==1
  	lombscargleOneObjectError<<< numBlocks, BLOCKSIZE>>>(dev_timeX, dev_magY, dev_magDY, dev_pgram, szData, minFreq, maxFreq, numFreqs);
  	#endif


  	//Find the index of the maximum power
  	thrust::device_ptr<DTYPE> maxLoc;
  	thrust::device_ptr<DTYPE> dev_ptr_pgram = thrust::device_pointer_cast(dev_pgram);

  	maxLoc = thrust::max_element(dev_ptr_pgram, dev_ptr_pgram+numFreqs); 
  	unsigned int maxPowerIdx= maxLoc - dev_ptr_pgram;
  	double freqStep=(maxFreq-minFreq)/(numFreqs*1.0);
  	//Validation: total period values
	*periodFound=(1.0/(minFreq+(maxPowerIdx*freqStep)))*2.0*M_PI;
	
  	printf("\nPeriod: %f", *periodFound);
  	cudaDeviceSynchronize();

  	double tend=omp_get_wtime();
  	printf("\nTime to compute kernel: %f",tend-tstart);

  	//copy pgram back to the host if enabled
  	#if RETURNPGRAM==1
  	*pgram=(DTYPE *)malloc(sizeof(DTYPE)*numFreqs);
  	#if PINNED==0
  	//standard if we don't use pinned memory for data transfers.
  	gpuErrchk(cudaMemcpy( *pgram, dev_pgram, sizeof(DTYPE)*numFreqs, cudaMemcpyDeviceToHost));
  	#endif

  	#if PINNED==1
  	DTYPE * pinned_buffer;
	unsigned int sizeBufferElems=(SIZEPINNEDBUFFERMIB*1024*1024)/(sizeof(DTYPE));
	gpuErrchk(cudaMallocHost((void**)&pinned_buffer, sizeof(DTYPE)*sizeBufferElems*NSTREAMS));	

  	unsigned int sizeDevData=numFreqs;
  	pinnedMemoryCopyDtoH(pinned_buffer, sizeBufferElems, dev_pgram, *pgram, sizeDevData);
	#endif

  	printf("\nMaximum power at found period: %f", (*pgram)[maxPowerIdx]);
	// fprintf(stderr,"Total elements transferred: %u",totalelemstransfered);

  	#endif


 //  	//pgram object
 //  	for (unsigned int i=0; i<numFreqs; i++)
	// {
	// 	printf("\n%d, %f",i, pgram[i]);
	// }
  	

  	//Output best periods to file
  	/*
  	char fnamebestperiods[]="bestperiods.txt";
  	printf("\nPrinting the best periods to file: %s", fnamebestperiods);
	ofstream bestperiodsoutput;
	bestperiodsoutput.open(fnamebestperiods,ios::out);	
  	bestperiodsoutput.precision(17);
  	for (unsigned int i=0; i<numUniqueObjects; i++)
	{
		bestperiodsoutput<<foundPeriod[i]<<endl;
	}
  	bestperiodsoutput.close();
	*/

  	//Output pgram to file

 //  	char fnameoutput[]="pgram.txt";
 //  	printf("\nPrinting the prgram to file: %s", fnameoutput);
	// ofstream pgramoutput;
	// pgramoutput.open(fnameoutput,ios::out);	
 //  	pgramoutput.precision(17);
 //  	for (unsigned int i=0; i<numFreqs; i++)
	// {
	// 	pgramoutput<<(*pgram)[i]<<endl;
	// }
 //  	pgramoutput.close();


	//Output pgram to file (small period range)
  	
 //  	char fnameoutput[]="pgram.txt";
 //  	printf("\nPrinting the prgram to file: %s", fnameoutput);
	// ofstream pgramoutput;
	// pgramoutput.open(fnameoutput,ios::out);	
	// pgramoutput<<"Period, power"<<endl;
 //  	pgramoutput.precision(17);
 //  	for (unsigned int i=0; i<numFreqs; i++)
	// {
	// 	double period=(2.0*M_PI/(minFreq+(i*freqStep)));
	// 	if (period>0.87 && period<0.881)
	// 	{
	// 	pgramoutput<<period<<", "<<(*pgram)[i]<<endl;
	// 	}
	// }
 //  	pgramoutput.close();
	




  	//free memory-- CUDA

  	cudaFree(dev_timeX);
  	cudaFree(dev_magY);
	cudaFree(dev_sizeData);
	cudaFree(dev_pgram);
	cudaFree(dev_foundPeriod);

	#if ERROR==1
	cudaFree(dev_magDY);
	#endif

	#if PINNED==1 && RETURNPGRAM==1
	cudaFreeHost(pinned_buffer);
	#endif
	
  	


}
























//Send the minimum and maximum frequency and number of frequencies to test to the GPU (not a list of frequencies)
void batchGPULS(unsigned int * objectId, DTYPE * timeX,  DTYPE * magY, DTYPE * magDY, unsigned int * sizeData, const DTYPE minFreq, const DTYPE maxFreq, const unsigned int numFreqs, DTYPE * sumPeriods, DTYPE ** pgram, DTYPE * foundPeriod)
{
	



	DTYPE * dev_timeX;
	DTYPE * dev_magY;
	unsigned int * dev_sizeData;
	DTYPE * dev_foundPeriod;
	DTYPE * dev_pgram;

	struct lookupObj * dev_objectLookup;

	//compute the object ranges in the arrays and store in struct
	//This is given by the objectId
	struct lookupObj * objectLookup=NULL;
	unsigned int numUniqueObjects;
	computeObjectRanges(objectId, sizeData, &objectLookup, &numUniqueObjects);

    foundPeriod=(DTYPE *)malloc(sizeof(DTYPE)*numUniqueObjects);


	//allocate memory on the GPU
	gpuErrchk(cudaMalloc((void**)&dev_timeX, sizeof(DTYPE)*(*sizeData)));
	gpuErrchk(cudaMalloc((void**)&dev_magY, sizeof(DTYPE)*(*sizeData)));

	//If astropy implementation with error
	#if ERROR==1
	DTYPE * dev_magDY;
	gpuErrchk(cudaMalloc((void**)&dev_magDY, sizeof(DTYPE)*(*sizeData)));

	//Need to incorporate error into magnitudes
	for (int i=0; i<numUniqueObjects; i++)
	{
		unsigned int idxMin=objectLookup[i].idxMin;
		unsigned int idxMax=objectLookup[i].idxMax;
		unsigned int sizeDataForObject=idxMax-idxMin+1;
		updateYerrorfactor(&magY[idxMin], &magDY[idxMin], sizeDataForObject);
	}	
	#endif
	
	#if RETURNPGRAM==1
	// Result periodogram must be number of unique objects * the size of the frequency array
	gpuErrchk(cudaMalloc((void**)&dev_pgram, sizeof(DTYPE)*numFreqs*numUniqueObjects));

	//Make a small pinned memory buffer for transferring the array back
	DTYPE * pinned_buffer;
	unsigned int sizeBufferElems=(SIZEPINNEDBUFFERMIB*1024*1024)/(sizeof(DTYPE));
	gpuErrchk(cudaMallocHost((void**)&pinned_buffer, sizeof(DTYPE)*sizeBufferElems*NSTREAMS));	
	#endif

	#if RETURNPGRAM==0
	//If not returning the pgram then do not allocate memory
	dev_pgram=NULL;
	#endif

	//the maximum power in each periodogram. Use this when we don't want to return the periodogram
	gpuErrchk(cudaMalloc((void**)&dev_foundPeriod, sizeof(DTYPE)*numUniqueObjects));
	gpuErrchk(cudaMalloc((void**)&dev_sizeData, sizeof(unsigned int)));
	gpuErrchk(cudaMalloc((void**)&dev_objectLookup, sizeof(lookupObj)*(numUniqueObjects)));

	//copy data to the GPU
	gpuErrchk(cudaMemcpy( dev_timeX, timeX, sizeof(DTYPE)*(*sizeData), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy( dev_magY, magY, sizeof(DTYPE)*(*sizeData), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy( dev_sizeData, sizeData, sizeof(unsigned int), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy( dev_objectLookup, objectLookup, sizeof(lookupObj)*(numUniqueObjects), cudaMemcpyHostToDevice));

	#if ERROR==1
	gpuErrchk(cudaMemcpy( dev_magDY, magDY, sizeof(DTYPE)*(*sizeData), cudaMemcpyHostToDevice));
	#endif

	
	const int numBlocks=numUniqueObjects;
	double tstart=omp_get_wtime();
  	//Do lomb-scargle
  	#if ERROR==0 && SHMEM==0
  	lombscargleBatch<<< numBlocks, BLOCKSIZE>>>(dev_timeX, dev_magY, dev_objectLookup, dev_pgram, dev_foundPeriod, minFreq, maxFreq, numFreqs);
  	#elif ERROR==0 && SHMEM==1
  	lombscargleBatchSM<<< numBlocks, BLOCKSIZE>>>(dev_timeX, dev_magY, dev_objectLookup, dev_pgram, dev_foundPeriod, minFreq, maxFreq, numFreqs);
  	#elif ERROR==1 //no SHMEM option
  	lombscargleBatchError<<< numBlocks, BLOCKSIZE>>>(dev_timeX, dev_magY, dev_magDY, dev_objectLookup, dev_pgram, dev_foundPeriod, minFreq, maxFreq, numFreqs);
  	#endif

  	cudaDeviceSynchronize();

  	double tend=omp_get_wtime();
  	printf("\nTime to compute kernel: %f",tend-tstart);


  	

  	//copy pgram back to the host if enabled
  	#if RETURNPGRAM==1
  	*pgram=(DTYPE *)malloc(sizeof(DTYPE)*numFreqs*numUniqueObjects);

  	#if PINNED==0
  	//standard if we don't use pinned memory for data transfers.
  	gpuErrchk(cudaMemcpy( *pgram, dev_pgram, sizeof(DTYPE)*numUniqueObjects*numFreqs, cudaMemcpyDeviceToHost));
  	#endif

  	#if PINNED==1
  	unsigned int sizeDevData=numUniqueObjects*numFreqs;
  	pinnedMemoryCopyDtoH(pinned_buffer, sizeBufferElems, dev_pgram, *pgram, sizeDevData);
	#endif

	// fprintf(stderr,"Total elements transferred: %u",totalelemstransfered);

  	#endif



  	//Return the maximum power for each object
  	#if RETURNPGRAM==0
  	gpuErrchk(cudaMemcpy( foundPeriod, dev_foundPeriod, sizeof(DTYPE)*(numUniqueObjects), cudaMemcpyDeviceToHost));
  	#endif
  	
  	

  	//For each object, find the maximum power in the pgram
  	//then find the corresponding line in the periods file from daniel

  	#if RETURNPGRAM==1
  	//compute the maximum power using the returned pgram
  	double tstartcpupgram=omp_get_wtime();
  	printf("\nCompute period from pgram on CPU:");

  	double freqStep=(maxFreq-minFreq)/(numFreqs*1.0);	
  	#pragma omp parallel for num_threads(NTHREADSCPU)
  	for (int i=0; i<numUniqueObjects; i++)
  	{
  		DTYPE maxPower=0;
  		unsigned int maxPowerIdx=0;
	  	for (int j=0; j<numFreqs; j++)
	  	{
	  		if (maxPower<(*pgram)[i*numFreqs+j])
	  		{
	  			maxPower=(*pgram)[i*numFreqs+j];
	  			maxPowerIdx=j;
	  		}
	  	}

	  	//Validation: total period values
		foundPeriod[i]=(1.0/(minFreq+(maxPowerIdx*freqStep)))*2.0*M_PI;

	  	
  	
	  	// if (i==0)
	  	// {
	  	// 	printf("\nmaxPowerIdx: %u", maxPowerIdx);
	  	// }
  	}



  	double tendcpupgram=omp_get_wtime();
  	printf("\nTime to compute the periods on the CPU using the pgram: %f", tendcpupgram - tstartcpupgram);
  	#endif	


  	#if RETURNPGRAM==0
  	printf("\nCompute period in the kernel directly bypassing the pgram array");
  	#endif

  	#if PRINTPERIODS==1
  	for (int i=0; i<numUniqueObjects; i++)
  	{
	  	printf("\nObject: %d Period: %f, ",objectLookup[i].objId,foundPeriod[i]);
  	}
  	#endif

  	//Validation: total period values
  	
  	for (int i=0; i<numUniqueObjects; i++)
  	{
	  	(*sumPeriods)+=foundPeriod[i];
  	}

  	//Output best periods to file
  	/*
  	char fnamebestperiods[]="bestperiods.txt";
  	printf("\nPrinting the best periods to file: %s", fnamebestperiods);
	ofstream bestperiodsoutput;
	bestperiodsoutput.open(fnamebestperiods,ios::out);	
  	bestperiodsoutput.precision(17);
  	for (unsigned int i=0; i<numUniqueObjects; i++)
	{
		bestperiodsoutput<<foundPeriod[i]<<endl;
	}
  	bestperiodsoutput.close();
	
  	//Output pgram to file
  	char fnameoutput[]="pgram.txt";
  	printf("\nPrinting the prgram to file: %s", fnameoutput);
	ofstream pgramoutput;
	pgramoutput.open(fnameoutput,ios::out);	
  	pgramoutput.precision(17);
  	for (unsigned int i=0; i<numUniqueObjects*numFreqs; i++)
	{
		pgramoutput<<pgram[i]<<endl;
	}
  	pgramoutput.close();
	*/

  	//free memory
  	free(foundPeriod);
  	free(objectLookup);

  	//free memory-- CUDA
  	cudaFree(dev_timeX);
  	cudaFree(dev_magY);
	cudaFree(dev_sizeData);
	cudaFree(dev_pgram);
	cudaFree(dev_foundPeriod);
	cudaFree(dev_objectLookup);

	#if ERROR==1
	cudaFree(dev_magDY); 
	#endif

	#if PINNED==1 && RETURNPGRAM==1
	cudaFreeHost(pinned_buffer);
	#endif

	
	


  	


}





//parallelize over the frequency if computing a single object
void lombscargleCPUOneObject(DTYPE * timeX,  DTYPE * magY, unsigned int * sizeData, const DTYPE minFreq, const DTYPE maxFreq, const unsigned int numFreqs, DTYPE * foundPeriod, DTYPE * pgram)
{
	pgram=(DTYPE *)malloc(sizeof(DTYPE)*(numFreqs));

	const DTYPE freqStep=(maxFreq-minFreq)/(numFreqs*1.0);	

	//1 refers to the mode of executing in parallel inside the LS algorithm
	lombscarglecpu(1, timeX, magY, *sizeData, numFreqs, minFreq, maxFreq, freqStep, pgram);	
	computePeriod(pgram, numFreqs, minFreq, freqStep, foundPeriod);

	#if PRINTPERIODS==1
	printf("\nPeriod: %f", *foundPeriod);
	#endif
}

//uses error propogation
void lombscargleCPUOneObjectError(DTYPE * timeX,  DTYPE * magY, DTYPE * magDY, unsigned int * sizeData, const DTYPE minFreq, const DTYPE maxFreq, const unsigned int numFreqs, DTYPE * foundPeriod, DTYPE * pgram)
{
	pgram=(DTYPE *)malloc(sizeof(DTYPE)*(numFreqs));

	const DTYPE freqStep=(maxFreq-minFreq)/(numFreqs*1.0);	

	//1 refers to the mode of executing in parallel inside the LS algorithm
	lombscarglecpuError(1, timeX, magY, magDY, *sizeData, numFreqs, minFreq, maxFreq, freqStep, pgram);	
	computePeriod(pgram, numFreqs, minFreq, freqStep, foundPeriod);

	#if PRINTPERIODS==1
	printf("\nPeriod: %f", *foundPeriod);
	#endif

	// char fnameoutput[]="pgram.txt";
 //  	printf("\nPrinting the prgram to file: %s", fnameoutput);
	// ofstream pgramoutput;
	// pgramoutput.open(fnameoutput,ios::out);	
 //  	pgramoutput.precision(17);
 //  	for (unsigned int i=0; i<numFreqs; i++)
	// {
	// 	pgramoutput<<pgram[i]<<endl;
	// }
 //  	pgramoutput.close();
		

}


void lombscargleCPUBatchError(unsigned int * objectId, DTYPE * timeX,  DTYPE * magY, DTYPE * magDY, unsigned int * sizeData, const DTYPE minFreq, const DTYPE maxFreq, const unsigned int numFreqs, DTYPE * sumPeriods, DTYPE * pgram, DTYPE * foundPeriod)
{


	//compute the object ranges in the arrays and store in struct
	//This is given by the objectId
	struct lookupObj * objectLookup=NULL;
	unsigned int numUniqueObjects;
	computeObjectRanges(objectId, sizeData, &objectLookup, &numUniqueObjects);
	pgram=(DTYPE *)malloc(sizeof(DTYPE)*(numFreqs)*numUniqueObjects);
	foundPeriod=(DTYPE *)malloc(sizeof(DTYPE)*numUniqueObjects);

	const DTYPE freqStep=(maxFreq-minFreq)/(numFreqs*1.0);	

	//for each object, call the sequential cpu algorithm
	#pragma omp parallel for num_threads(NTHREADSCPU) schedule(dynamic)
	for (int i=0; i<numUniqueObjects; i++)
	{
		unsigned int idxMin=objectLookup[i].idxMin;
		unsigned int idxMax=objectLookup[i].idxMax;
		unsigned int sizeDataForObject=idxMax-idxMin+1;
		//0 refers to the mode of executing sequentially inside the LS algorithm
		lombscarglecpuError(0, &timeX[idxMin], &magY[idxMin], &magDY[idxMin], sizeDataForObject, numFreqs, minFreq, maxFreq, freqStep, pgram+(i*numFreqs));	
		computePeriod(pgram+(i*numFreqs), numFreqs, minFreq, freqStep, &foundPeriod[i]);
	}

	#if PRINTPERIODS==1
	for (int i=0; i<numUniqueObjects; i++)
	{
	printf("\nObject: %d, Period: %f",objectLookup[i].objId, foundPeriod[i]);
	}
	#endif



	//Validation
 	for (int i=0; i<numUniqueObjects; i++)
  	{
	  	(*sumPeriods)+=foundPeriod[i];
  	}

	

}


void lombscargleCPUBatch(unsigned int * objectId, DTYPE * timeX,  DTYPE * magY, unsigned int * sizeData, const DTYPE minFreq, const DTYPE maxFreq, const unsigned int numFreqs, DTYPE * sumPeriods, DTYPE * pgram, DTYPE * foundPeriod)
{


	//compute the object ranges in the arrays and store in struct
	//This is given by the objectId
	struct lookupObj * objectLookup=NULL;
	unsigned int numUniqueObjects;
	computeObjectRanges(objectId, sizeData, &objectLookup, &numUniqueObjects);
	pgram=(DTYPE *)malloc(sizeof(DTYPE)*(numFreqs)*numUniqueObjects);
	foundPeriod=(DTYPE *)malloc(sizeof(DTYPE)*numUniqueObjects);

	const DTYPE freqStep=(maxFreq-minFreq)/(numFreqs*1.0);	

	//for each object, call the sequential cpu algorithm
	#pragma omp parallel for num_threads(NTHREADSCPU) schedule(dynamic)
	for (int i=0; i<numUniqueObjects; i++)
	{
		unsigned int idxMin=objectLookup[i].idxMin;
		unsigned int idxMax=objectLookup[i].idxMax;
		unsigned int sizeDataForObject=idxMax-idxMin+1;
		//0 refers to the mode of executing sequentially inside the LS algorithm
		lombscarglecpu(0, &timeX[idxMin], &magY[idxMin], sizeDataForObject, numFreqs, minFreq, maxFreq, freqStep, pgram+(i*numFreqs));	
		computePeriod(pgram+(i*numFreqs), numFreqs, minFreq, freqStep, &foundPeriod[i]);
	}

	#if PRINTPERIODS==1
	for (int i=0; i<numUniqueObjects; i++)
	{
	printf("\nObject: %d, Period: %f",objectLookup[i].objId, foundPeriod[i]);
	}
	#endif



	//Validation
 	for (int i=0; i<numUniqueObjects; i++)
  	{
	  	(*sumPeriods)+=foundPeriod[i];
  	}

	

}

void computePeriod(DTYPE * pgram, const unsigned int numFreqs, const DTYPE minFreq, const DTYPE freqStep, DTYPE * foundPeriod)
{
		DTYPE maxPower=0;
  		unsigned int maxPowerIdx=0;
	  	for (int i=0; i<numFreqs; i++)
	  	{
	  		if (maxPower<pgram[i])
	  		{
	  			maxPower=pgram[i];
	  			maxPowerIdx=i;
	  		}
	  	}

	  	// printf("\nMax power idx: %d", maxPowerIdx);
	  	
	  	//Validation: total period values
		*foundPeriod=(1.0/(minFreq+(maxPowerIdx*freqStep)))*2.0*M_PI;
	  	
}


void lombscarglecpuinnerloop(int iteration, DTYPE * x, DTYPE * y, DTYPE * pgram, DTYPE * freqToTest, const unsigned int sizeData)
{
			DTYPE c, s, xc, xs, cc, ss, cs;
	    	DTYPE tau, c_tau, s_tau, c_tau2, s_tau2, cs_tau;
	        xc = 0.0;
	        xs = 0.0;
	        cc = 0.0;
	        ss = 0.0;
	        cs = 0.0;

	        for (unsigned int j=0; j<sizeData; j++)
	        {	
	            c = cos((*freqToTest) * x[j]);
	            s = sin((*freqToTest) * x[j]);

	            xc += y[j] * c;
	            xs += y[j] * s;
	            cc += c * c;
	            ss += s * s;
	            cs += c * s;
	        }
	            
	        tau = atan2(2.0 * cs, cc - ss) / (2.0 * (*freqToTest));
	        c_tau = cos((*freqToTest) * tau);
	        s_tau = sin((*freqToTest) * tau);
	        c_tau2 = c_tau * c_tau;
	        s_tau2 = s_tau * s_tau;
	        cs_tau = 2.0 * c_tau * s_tau;

	        pgram[iteration] = 0.5 * ((((c_tau * xc + s_tau * xs)*(c_tau * xc + s_tau * xs)) / 
	            (c_tau2 * cc + cs_tau * cs + s_tau2 * ss)) + 
	            (((c_tau * xs - s_tau * xc)*(c_tau * xs - s_tau * xc)) / 
	            (c_tau2 * ss - cs_tau * cs + s_tau2 * cc)));
}

//lombsscarge on the CPU 
//Mode==0 means run sequentially
//Mode==1 means parallelize over the frequency loop
void lombscarglecpu(bool mode, DTYPE * x, DTYPE * y, const unsigned int sizeData, const unsigned int numFreqs, const DTYPE minFreq, const DTYPE maxFreq, const DTYPE freqStep, DTYPE * pgram)
{

	    if (mode==0)
	    {	
			for (int i=0; i<numFreqs; i++)
			{
				DTYPE freqToTest=minFreq+(freqStep*i);
				lombscarglecpuinnerloop(i, x, y, pgram, &freqToTest, sizeData);
		    }
		}
		else if(mode==1)
		{
			#pragma omp parallel for num_threads(NTHREADSCPU) schedule(static)
			for (int i=0; i<numFreqs; i++)
			{
				DTYPE freqToTest=minFreq+(freqStep*i);
				lombscarglecpuinnerloop(i, x, y, pgram, &freqToTest, sizeData);
		    }
		}

}


//Pre-center the data with error:		
// w = dy ** -2
//    y = y - np.dot(w, y) / np.sum(w)
void updateYerrorfactor(DTYPE * y, DTYPE *dy, const unsigned int sizeData)
{

		//Pre-center the data with error:
		//w = dy ** -2
		//sum w
		DTYPE * w =(DTYPE *)malloc(sizeof(DTYPE)*sizeData);
		DTYPE sumw=0;
		#pragma omp parallel for num_threads(NTHREADSCPU) reduction(+:sumw)
		for (int i=0; i<sizeData; i++)
		{
			w[i]=1.0/sqrt(dy[i]);
			sumw+=w[i];
		}
		//compute dot product w,y
		DTYPE dotwy=0;
		#pragma omp parallel for num_threads(NTHREADSCPU) reduction(+:dotwy)
		for (int i=0; i<sizeData; i++)
		{
			dotwy+=w[i]*y[i];
		}

		//update y to account for dot product and sum w
		//y = y - dot(w, y) / np.sum(w)	
		#pragma omp parallel for num_threads(NTHREADSCPU)
		for (int i=0; i<sizeData; i++)
		{
			y[i]=y[i]-dotwy/sumw;
		}

		free(w);
}

//lombsscarge on the CPU for AstroPy with error
//Mode==0 means run sequentially (batch mode)
//Mode==1 means parallelize over the frequency loop (multiobject)
void lombscarglecpuError(bool mode, DTYPE * x, DTYPE * y, DTYPE *dy, const unsigned int sizeData, const unsigned int numFreqs, const DTYPE minFreq, const DTYPE maxFreq, const DTYPE freqStep, DTYPE * pgram)
{
		// printf("\nExecuting astropy version");


		updateYerrorfactor(y, dy, sizeData);


	    if (mode==0)
	    {	
			for (int i=0; i<numFreqs; i++)
			{
				DTYPE freqToTest=minFreq+(freqStep*i);
				lombscarglecpuinnerloopAstroPy(i, x, y, dy, pgram, &freqToTest, sizeData);
		    }
		}
		else if(mode==1)
		{
			
			#pragma omp parallel for num_threads(NTHREADSCPU) schedule(static)
			for (int i=0; i<numFreqs; i++)
			{
				DTYPE freqToTest=minFreq+(freqStep*i);
				lombscarglecpuinnerloopAstroPy(i, x, y, dy, pgram, &freqToTest, sizeData);
		    }
		}

}


//AstroPy has error propogration and fits to the mean
//Ported from here:
//https://github.com/astropy/astropy/blob/master/astropy/timeseries/periodograms/lombscargle/implementations/cython_impl.pyx
//Uses the generalized periodogram in the cython code
void lombscarglecpuinnerloopAstroPy(int iteration, DTYPE * x, DTYPE * y, DTYPE * dy, DTYPE * pgram, 
	DTYPE * freqToTest, const unsigned int sizeData)
{

	DTYPE w, omega_t, sin_omega_t, cos_omega_t, S, C, S2, C2, tau, Y, wsum, YY, Stau, Ctau, YCtau, YStau, CCtau, SStau; 

	wsum = 0.0;
	S = 0.0;
	C = 0.0;
	S2 = 0.0;
	C2 = 0.0;

	//first pass: determine tau
	for (int j=0; j<sizeData; j++)
	{
    w = 1.0 / dy[j];
    w *= w;
    wsum += w;
    omega_t = (*freqToTest) * x[j];
    sin_omega_t = sin(omega_t);
    cos_omega_t = cos(omega_t);
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
		tau = 0.5 * atan2(S2, C2) / (*freqToTest);
		Y = 0.0;
		YY = 0.0;
		Stau = 0.0;
		Ctau = 0.0;
		YCtau = 0.0;
		YStau = 0.0;
		CCtau = 0.0;
		SStau = 0.0;
		// second pass: compute the power
		for (int j=0; j<sizeData; j++)
		{
		    w = 1.0 / dy[j];
		    w *= w;
		    omega_t = (*freqToTest) * (x[j] - tau);
		    sin_omega_t = sin(omega_t);
		    cos_omega_t = cos(omega_t);
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
		


    pgram[iteration] = (YCtau * YCtau / CCtau + YStau * YStau / SStau) / YY;
}



void importObjXYData(char * fnamedata, unsigned int * sizeData, unsigned int ** objectId, DTYPE ** timeX, DTYPE ** magY, DTYPE ** magDY)
{

	//import objectId, timeX, magY

	std::vector<DTYPE>tmpAllData;
	std::ifstream in(fnamedata);
	unsigned int cnt=0;
	for (std::string f; getline(in, f, ',');){

	DTYPE i;
		 std::stringstream ss(f);
	    while (ss >> i)
	    {
	        tmpAllData.push_back(i);
	        // array[cnt]=i;
	        cnt++;
	        if (ss.peek() == ',')
	            ss.ignore();
	    }

  	}




  	
  	#if ERROR==0
  	*sizeData=(unsigned int)tmpAllData.size()/3;
  	#endif

  	#if ERROR==1
  	*sizeData=(unsigned int)tmpAllData.size()/4;
  	#endif
  	printf("\nData import: Total rows: %u",*sizeData);
  	
  	*objectId=(unsigned int *)malloc(sizeof(DTYPE)*(*sizeData));
  	*timeX=   (DTYPE *)malloc(sizeof(DTYPE)*(*sizeData));
  	*magY=    (DTYPE *)malloc(sizeof(DTYPE)*(*sizeData));

  	#if ERROR==1
  	*magDY=    (DTYPE *)malloc(sizeof(DTYPE)*(*sizeData));
  	#endif


  	#if ERROR==0
  	for (int i=0; i<*sizeData; i++){
  		(*objectId)[i]=tmpAllData[(i*3)+0];
  		(*timeX)[i]   =tmpAllData[(i*3)+1];
  		(*magY)[i]    =tmpAllData[(i*3)+2];
  	}
  	#endif

  	#if ERROR==1
  	for (int i=0; i<*sizeData; i++){
  		(*objectId)[i]=tmpAllData[(i*4)+0];
  		(*timeX)[i]   =tmpAllData[(i*4)+1];
  		(*magY)[i]    =tmpAllData[(i*4)+2];
  		(*magDY)[i]    =tmpAllData[(i*4)+3];
  	}
  	#endif

}




void warmUpGPU(){
printf("\nLoad CUDA runtime (initialization overhead)\n");
cudaDeviceSynchronize();
}