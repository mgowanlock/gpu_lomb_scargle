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


//GPU constants, do not change
#define BLOCKSIZE 512 //CUDA block size
#define SIZEPINNEDBUFFERMIB 8 //When returning the pgram, need to transfer using pinned memory. Use this many MiB per buffer
#define NSTREAMS 3 //When returning the pgram use this many streams each of size SIZEPINNEDBUFFERMIB


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
void importObjXYData(char * fnamedata, unsigned int * sizeData, unsigned int ** objectId, DTYPE ** timeX, DTYPE ** magY);
int detectMode(unsigned int * objectId, unsigned int * sizeData);
void batchGPULS(unsigned int * objectId, DTYPE * timeX,  DTYPE * magY, unsigned int * sizeData, const DTYPE minFreq, const DTYPE maxFreq, const unsigned int numFreqs, DTYPE * sumPeriods, DTYPE ** pgram, DTYPE * foundPeriod);
void GPULSOneObject(DTYPE * timeX,  DTYPE * magY, unsigned int * sizeData, const DTYPE minFreq, const DTYPE maxFreq, const unsigned int numFreqs, DTYPE * periodFound, DTYPE ** pgram);
void computeObjectRanges(unsigned int * objectId, unsigned int * sizeData, struct lookupObj ** objectLookup, unsigned int * numUniqueObjects);
void pinnedMemoryCopyDtoH(DTYPE * pinned_buffer, unsigned int sizeBufferElems, DTYPE * dev_data, DTYPE * pageable, unsigned int sizeTotalData);
void warmUpGPU();


using namespace std;

int main(int argc, char *argv[])
{

	warmUpGPU();
	omp_set_nested(1);


	/////////////////////////
	// Get information from command line
	/////////////////////////

	
	//Input data filename (objId, time, amplitude), minimum frequency, maximum frequency, numer of frequencies, mode
	if (argc!=5)
	{
	cout <<"\n\nIncorrect number of input parameters.\nExpected values: Data filename (objId, time, amplitude), minimum frequency, maximum frequency, number of frequencies\n";
	return 0;
	}
	
	
	char inputFname[500];
	strcpy(inputFname,argv[1]);
	const double minFreq=atof(argv[2]); //inclusive
	const double maxFreq=atof(argv[3]); //exclusive
	const unsigned int freqToTest=atoi(argv[4]);
    

	printf("\nDataset file: %s",inputFname);
	printf("\nMinimum Frequency: %f",minFreq);
	printf("\nMaximum Frequency: %f",maxFreq);
	printf("\nNumber of frequencies to test: %u", freqToTest);
	
	unsigned int * objectId=NULL; 
	DTYPE * timeX=NULL; 
	DTYPE * magY=NULL;
	unsigned int sizeData;
	importObjXYData(inputFname, &sizeData, &objectId, &timeX, &magY);	

	//determine whether file has a single object or multiple objects
	int MODE=detectMode(objectId, &sizeData);
	
	//pgram allocated in the functions below
	//Stores the LS power for each frequency
	DTYPE * pgram=NULL;

	//foundPeriod is allocated in the functions below
	DTYPE * foundPeriod=NULL;

	//Batch of LS to compute on the GPU
	if (MODE==1)
	{
		printf("\nMode: %d Detected [Processing a batch of objects]", MODE);
		DTYPE sumPeriods=0;
		
		double tstart=omp_get_wtime();
		
		batchGPULS(objectId, timeX, magY, &sizeData, minFreq, maxFreq, freqToTest, &sumPeriods, &pgram, foundPeriod);
		
		double tend=omp_get_wtime();
		double totalTime=tend-tstart;
		printf("\nTotal time to compute batch: %f", totalTime);
		printf("\n[Validation] Sum of all periods: %f", sumPeriods);

	}
	//One LS to compute on the GPU
	else if (MODE==2)
	{
		printf("\nMode: %d Detected [Processing a single object]", MODE);
		DTYPE periodFound=0;	
		double tstart=omp_get_wtime();
		
		GPULSOneObject(timeX, magY, &sizeData, minFreq, maxFreq, freqToTest, &periodFound, &pgram);
	
		double tend=omp_get_wtime();
		double totalTime=tend-tstart;
		printf("\nTotal time to compute batch: %f", totalTime);
		printf("\n[Validation] Period: %f", periodFound);
	}
	

	//free memory
	free(foundPeriod);
	free(objectId);
	free(timeX);
	free(magY);
	free(pgram);

	printf("\n");
	return 0;
}

//return 1 if there's multiple objects in the file
//return 2 if there's a single object in the file
int detectMode(unsigned int * objectId, unsigned int * sizeData)
{
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

	

	if (cntUnique>1){
		return 1;
	}
	else{
		return 2;
	}
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
void GPULSOneObject(DTYPE * timeX,  DTYPE * magY, unsigned int * sizeData, const DTYPE minFreq, const DTYPE maxFreq, const unsigned int numFreqs, DTYPE * periodFound, DTYPE ** pgram)
{
	

	DTYPE * dev_timeX;
	DTYPE * dev_magY;
	unsigned int * dev_sizeData;
	DTYPE * dev_pgram;
	DTYPE * dev_foundPeriod;

	//allocate memory on the GPU
	gpuErrchk(cudaMalloc((void**)&dev_timeX, sizeof(DTYPE)*(*sizeData)));
	gpuErrchk(cudaMalloc((void**)&dev_magY, sizeof(DTYPE)*(*sizeData)));
	
	
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

	const unsigned int szData=*sizeData;
	const unsigned int numBlocks=ceil(numFreqs*1.0/BLOCKSIZE*1.0);
	
	double tstart=omp_get_wtime();
  	//Do lomb-scargle
  	
  	lombscargleOneObject<<< numBlocks, BLOCKSIZE>>>(dev_timeX, dev_magY, dev_pgram, szData, minFreq, maxFreq, numFreqs);
  	

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

  	//copy pgram back to the host
  	*pgram=(DTYPE *)malloc(sizeof(DTYPE)*numFreqs);
  	//copy using pinned memory
  	DTYPE * pinned_buffer;
	unsigned int sizeBufferElems=(SIZEPINNEDBUFFERMIB*1024*1024)/(sizeof(DTYPE));
	gpuErrchk(cudaMallocHost((void**)&pinned_buffer, sizeof(DTYPE)*sizeBufferElems*NSTREAMS));	

  	unsigned int sizeDevData=numFreqs;
  	pinnedMemoryCopyDtoH(pinned_buffer, sizeBufferElems, dev_pgram, *pgram, sizeDevData);

  	//free memory-- CUDA
  	cudaFree(dev_timeX);
  	cudaFree(dev_timeX);
  	cudaFree(dev_magY);
	cudaFree(dev_sizeData);
	cudaFree(dev_pgram);
	cudaFree(dev_foundPeriod);
	cudaFreeHost(pinned_buffer);
	
}




//Send the minimum and maximum frequency and number of frequencies to test to the GPU (not a list of frequencies)
void batchGPULS(unsigned int * objectId, DTYPE * timeX,  DTYPE * magY, unsigned int * sizeData, const DTYPE minFreq, const DTYPE maxFreq, const unsigned int numFreqs, DTYPE * sumPeriods, DTYPE ** pgram, DTYPE * foundPeriod)
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
	
	//Allocate pgram
	// Result periodogram must be number of unique objects * the size of the frequency array
	gpuErrchk(cudaMalloc((void**)&dev_pgram, sizeof(DTYPE)*numFreqs*numUniqueObjects));

	//Make a small pinned memory buffer for transferring the array back
	DTYPE * pinned_buffer;
	unsigned int sizeBufferElems=(SIZEPINNEDBUFFERMIB*1024*1024)/(sizeof(DTYPE));
	gpuErrchk(cudaMallocHost((void**)&pinned_buffer, sizeof(DTYPE)*sizeBufferElems*NSTREAMS));	

	//the maximum power in each periodogram. Use this when we don't want to return the periodogram
	gpuErrchk(cudaMalloc((void**)&dev_foundPeriod, sizeof(DTYPE)*numUniqueObjects));
	gpuErrchk(cudaMalloc((void**)&dev_sizeData, sizeof(unsigned int)));
	gpuErrchk(cudaMalloc((void**)&dev_objectLookup, sizeof(lookupObj)*(numUniqueObjects)));

	//copy data to the GPU
	gpuErrchk(cudaMemcpy( dev_timeX, timeX, sizeof(DTYPE)*(*sizeData), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy( dev_magY, magY, sizeof(DTYPE)*(*sizeData), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy( dev_sizeData, sizeData, sizeof(unsigned int), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy( dev_objectLookup, objectLookup, sizeof(lookupObj)*(numUniqueObjects), cudaMemcpyHostToDevice));

	const int numBlocks=numUniqueObjects;
	double tstart=omp_get_wtime();
  	//Do lomb-scargle
  	
  	lombscargleBatch<<< numBlocks, BLOCKSIZE>>>(dev_timeX, dev_magY, dev_objectLookup, dev_pgram, dev_foundPeriod, minFreq, maxFreq, numFreqs);
  	

  	cudaDeviceSynchronize();

  	double tend=omp_get_wtime();
  	printf("\nTime to compute kernel: %f",tend-tstart);


  	

  	//copy pgram back to the host
  	*pgram=(DTYPE *)malloc(sizeof(DTYPE)*numFreqs*numUniqueObjects);
  	//copy pgram back using pinned memory
  	unsigned int sizeDevData=numUniqueObjects*numFreqs;
  	pinnedMemoryCopyDtoH(pinned_buffer, sizeBufferElems, dev_pgram, *pgram, sizeDevData);

  	//For each object, find the maximum power in the pgram
  	//then find the corresponding line in the periods file from daniel
  	
  	//compute the maximum power using the returned pgram
  	double tstartcpupgram=omp_get_wtime();
  	printf("\nCompute period from pgram on CPU:");

  	double freqStep=(maxFreq-minFreq)/(numFreqs*1.0);	
  	#pragma omp parallel for num_threads(4)
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

	  	foundPeriod[i]=(1.0/(minFreq+(maxPowerIdx*freqStep)))*2.0*M_PI;
  	
	  	// if (i==0)
	  	// {
	  	// 	printf("\nmaxPowerIdx: %u", maxPowerIdx);
	  	// }
  	}



  	double tendcpupgram=omp_get_wtime();
  	printf("\nTime to compute the periods on the CPU using the pgram: %f", tendcpupgram - tstartcpupgram);
  	

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

  	//free memory
  	free(foundPeriod);
  	free(objectLookup);

  	//free memory-- CUDA
  	cudaFree(dev_timeX);
  	cudaFree(dev_timeX);
  	cudaFree(dev_magY);
	cudaFree(dev_sizeData);
	cudaFree(dev_pgram);
	cudaFree(dev_foundPeriod);
	cudaFree(dev_objectLookup);
	cudaFreeHost(pinned_buffer);
	

}



void importObjXYData(char * fnamedata, unsigned int * sizeData, unsigned int ** objectId, DTYPE ** timeX, DTYPE ** magY)
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


  	*sizeData=(unsigned int)tmpAllData.size()/3;
  	printf("\nData import: Total rows: %u",*sizeData);
  	
  	*objectId=(unsigned int *)malloc(sizeof(DTYPE)*(*sizeData));
  	*timeX=   (DTYPE *)malloc(sizeof(DTYPE)*(*sizeData));
  	*magY=    (DTYPE *)malloc(sizeof(DTYPE)*(*sizeData));



  	for (int i=0; i<*sizeData; i++){
  		(*objectId)[i]=tmpAllData[(i*3)+0];
  		(*timeX)[i]   =tmpAllData[(i*3)+1];
  		(*magY)[i]    =tmpAllData[(i*3)+2];
  	}

}


void warmUpGPU(){
printf("\nLoad CUDA runtime (initialization overhead)\n");
cudaDeviceSynchronize();
}