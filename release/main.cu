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
#include "kernel.h"



//GPU constants, do not change
#define BLOCKSIZE 512 //CUDA block size
#define SIZEPINNEDBUFFERMIB 8 //When returning the pgram, need to transfer using pinned memory. Use this many MiB per buffer
#define NSTREAMS 3 //When returning the pgram use this many streams each of size SIZEPINNEDBUFFERMIB

//Global Variables for modes
bool doubleMode;
bool printMode;
bool errorMode;

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

void warmUpGPU(){
	printf("\nLoad CUDA runtime (initialization overhead)\n");
	cudaDeviceSynchronize();
}

//function prototypes
int detectMode(unsigned int * objectId, unsigned int * sizeData);

//Flaot function prototypes
void importObjXYData_float(char * fnamedata, unsigned int * sizeData, unsigned int ** objectId, float ** timeX, float ** magY, float ** magDY);
void batchGPULS_float(unsigned int * objectId, float * timeX,  float * magY, float * magDY, unsigned int * sizeData, const float minFreq, const float maxFreq, const unsigned int numFreqs, float * sumPeriods, float ** pgram, float * foundPeriod);
void GPULSOneObject_float(float * timeX,  float * magY, float * magDY, unsigned int * sizeData, const float minFreq, const float maxFreq, const unsigned int numFreqs, float * periodFound, float ** pgram);
void computeObjectRanges_float(unsigned int * objectId, unsigned int * sizeData, struct lookupObj ** objectLookup, unsigned int * numUniqueObjects);
void pinnedMemoryCopyDtoH_float(float * pinned_buffer, unsigned int sizeBufferElems, float * dev_data, float * pageable, unsigned int sizeTotalData);
void updateYerrorfactor_float(float * y, float *dy, const unsigned int sizeData);


//Double function prototypes
void importObjXYData_double(char * fnamedata, unsigned int * sizeData, unsigned int ** objectId, double ** timeX, double ** magY, double ** magDY);
void batchGPULS_double(unsigned int * objectId, double * timeX,  double * magY, double * magDY, unsigned int * sizeData, const double minFreq, const double maxFreq, const unsigned int numFreqs, double * sumPeriods, double ** pgram, double * foundPeriod);
void GPULSOneObject_double(double * timeX,  double * magY, double * magDY, unsigned int * sizeData, const double minFreq, const double maxFreq, const unsigned int numFreqs, double * periodFound, double ** pgram);
void computeObjectRanges_double(unsigned int * objectId, unsigned int * sizeData, struct lookupObj ** objectLookup, unsigned int * numUniqueObjects);
void pinnedMemoryCopyDtoH_double(double * pinned_buffer, unsigned int sizeBufferElems, double * dev_data, double * pageable, unsigned int sizeTotalData);
void updateYerrorfactor_double(double * y, double *dy, const unsigned int sizeData);



using namespace std;

int main(int argc, char *argv[])
{

	warmUpGPU();
	omp_set_nested(1);


	/////////////////////////
	// Get information from command line
	/////////////////////////

	
	//Input data filename (objId, time, amplitude), minimum frequency, maximum frequency, numer of frequencies, mode
	if (argc!=8)
	{
	cout <<"\n\nIncorrect number of input parameters.\nExpected values: Data filename (objId, time, amplitude), minimum frequency, maximum frequency, number of frequencies, data type (0=float, 1=double), print mode (1=on, 0=off), Error Mode (1=on, 0 = off)\n";
	return 0;
	}
	
	
	char inputFname[500];
	strcpy(inputFname,argv[1]);
	doubleMode = (bool)atoi(argv[5]);
	printMode = (bool)atoi(argv[6]);
	errorMode = (bool)atoi(argv[7]);

	const unsigned int freqToTest=atoi(argv[4]);


	printf("\nDataset file: %s",inputFname);
	printf("\nMinimum Frequency: %f",atof(argv[2]));
	printf("\nMaximum Frequency: %f",atof(argv[3]));
	printf("\nNumber of frequencies to test: %u\n", freqToTest);
	if(doubleMode){
		printf("\nRunning with data type of double for better precision at cost of performance.");
		printf("\nSet argument 5 to 0 to increase performance at cost of accuracy\n");
	}else{
		printf("\nRunning with data type of float for better performance at cost of accuracy.");
		printf("\nSet argument 5 to 1 to increase accuracy at cost of performance\n");
	}	
	if(printMode){
		printf("\nPrinting is turned on. This will print the period of every object to screen.");
		printf("\nIt is recomended to turn this off for large datasets. Set argument 6 to 0 to disable printing\n");
	}else{
		printf("\nPrinting is turned off. To print the period of every object to screen, set argument 6 to 1.\n");
	}
	if(errorMode){
		printf("\nExecuting L-S variant from AstroPy that propogates error and floats the mean.");
		printf("\nThis can be disabled by setting argument 7 to 0.\n");
	}else{
		printf("\nExecuting L-S without errors, to turn on; set argument 7 to 1\n");
	}
	
	unsigned int * objectId=NULL; 
	unsigned int sizeData;	
	double tstart=omp_get_wtime();

	if(doubleMode){

		const double minFreq=atof(argv[2]); //inclusive
		const double maxFreq=atof(argv[3]); //exclusive

		double * timeX=NULL; 
		double * magY=NULL;
		double * magDY=NULL;
		importObjXYData_double(inputFname, &sizeData, &objectId, &timeX, &magY, &magDY);	

		//determine whether file has a single object or multiple objects
		int MODE=detectMode(objectId, &sizeData);

		//Batch of LS to compute on the GPU
		if (MODE==1)
		{
			printf("\nMode: %d Detected [Processing a batch of objects]", MODE);
		
			//Stores the LS power for each frequency
			double * pgram=NULL;
			//foundPeriod is allocated in the functions below
			double * foundPeriod=NULL;

			double sumPeriods=0;
			batchGPULS_double(objectId, timeX, magY, magDY, &sizeData, minFreq, maxFreq, freqToTest, &sumPeriods, &pgram, foundPeriod);

			double tend=omp_get_wtime();
			double totalTime=tend-tstart;
			printf("\nTotal time to compute batch: %f", totalTime);
			printf("\n[Validation] Sum of all periods: %f", sumPeriods);

			//free memory
			free(foundPeriod);

		}else if (MODE==2){ //One LS to compute on the GPU 

			//Stores the LS power for each frequency
			double * pgram=NULL;

			
			double periodFound=0;
			if(errorMode){
				updateYerrorfactor_double(magY, magDY, sizeData);	
			}
		
			GPULSOneObject_double(timeX, magY, magDY, &sizeData, minFreq, maxFreq, freqToTest, &periodFound, &pgram);

			double tend=omp_get_wtime();
			double totalTime=tend-tstart;
			printf("\nTotal time to compute period: %f", totalTime);
			printf("\n[Validation] Period: %f", periodFound);
			
			//free memory
			free(pgram);
		}

		//free memory
		free(objectId);
		free(timeX);
		free(magY);
		free(magDY);

	}else{ //for floats
		const float minFreq=atof(argv[2]); //inclusive
		const float maxFreq=atof(argv[3]); //exclusive

		float * timeX=NULL; 
		float * magY=NULL;
		float * magDY=NULL;
		importObjXYData_float(inputFname, &sizeData, &objectId, &timeX, &magY, &magDY);	

		//determine whether file has a single object or multiple objects
		int MODE=detectMode(objectId, &sizeData);

		//Batch of LS to compute on the GPU
		if (MODE==1)
		{
			printf("\nMode: %d Detected [Processing a batch of objects]", MODE);
		
			//Stores the LS power for each frequency
			float * pgram=NULL;
	
			//foundPeriod is allocated in the functions below
			float * foundPeriod=NULL;

			float sumPeriods=0;
			batchGPULS_float(objectId, timeX, magY, magDY, &sizeData, minFreq, maxFreq, freqToTest, &sumPeriods, &pgram, foundPeriod);
		
			double tend=omp_get_wtime();
			double totalTime=tend-tstart;
			printf("\nTotal time to compute batch: %f", totalTime);
			printf("\n[Validation] Sum of all periods: %f", sumPeriods);

			//free memory
			free(foundPeriod);
	
		}else if(MODE==2){ //compute a single object

			//Stores the LS power for each frequency
			float * pgram=NULL;

			float periodFound=0;
			if(errorMode){
				updateYerrorfactor_float(magY, magDY, sizeData);	
			}
			
			GPULSOneObject_float(timeX, magY, magDY, &sizeData, minFreq, maxFreq, freqToTest, &periodFound, &pgram);

			double tend=omp_get_wtime();
			double totalTime=tend-tstart;
			printf("\nTotal time to compute period: %f", totalTime);
			printf("\n[Validation] Period: %f", periodFound);

			//free memory
			free(pgram);
		}

		//free memory
		free(objectId);
		free(timeX);
		free(magY);
		free(magDY);
	}


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

//Float Section

void pinnedMemoryCopyDtoH_float(float * pinned_buffer, unsigned int sizeBufferElems, float * dev_data, float * pageable, unsigned int sizeTotalData)
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
		gpuErrchk(cudaMemcpyAsync(pinned_buffer+pinnedBufferOffset, dev_data+(offsetstart), sizeof(float)*elemsToTransfer, cudaMemcpyDeviceToHost, streams[tid])); 

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
void GPULSOneObject_float(float * timeX,  float * magY, float * magDY, unsigned int * sizeData, const float minFreq, const float maxFreq, const unsigned int numFreqs, float * periodFound, float ** pgram)
{
	

	float * dev_timeX;
	float * dev_magY;
	unsigned int * dev_sizeData;
	float * dev_pgram;
	float * dev_foundPeriod;

	//allocate memory on the GPU
	gpuErrchk(cudaMalloc((void**)&dev_timeX, sizeof(float)*(*sizeData)));
	gpuErrchk(cudaMalloc((void**)&dev_magY, sizeof(float)*(*sizeData)));
	

	
	// Result periodogram
	//need to allocate it on the GPUeven if we do not return it to the host so that we can find the maximum power
	gpuErrchk(cudaMalloc((void**)&dev_pgram, sizeof(float)*numFreqs));
	
	//the maximum power in the periodogram. Use this when we don't want to return the periodogram
	gpuErrchk(cudaMalloc((void**)&dev_foundPeriod, sizeof(float)));
	gpuErrchk(cudaMalloc((void**)&dev_sizeData, sizeof(unsigned int)));

	//copy data to the GPU
	gpuErrchk(cudaMemcpy( dev_timeX, timeX, sizeof(float)*(*sizeData), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy( dev_magY, magY, sizeof(float)*(*sizeData), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy( dev_sizeData, sizeData, sizeof(unsigned int), cudaMemcpyHostToDevice));


	const unsigned int szData=*sizeData;
	const unsigned int numBlocks=ceil(numFreqs*1.0/BLOCKSIZE*1.0);
	
	double tstart;
  	//Do lomb-scargle
	if(errorMode){
		float * dev_magDY;
		gpuErrchk(cudaMalloc((void**)&dev_magDY, sizeof(float)*(*sizeData)));
		gpuErrchk(cudaMemcpy( dev_magDY, magDY, sizeof(float)*(*sizeData), cudaMemcpyHostToDevice));

		tstart = omp_get_wtime();
		lombscargleOneObjectError_float<<< numBlocks, BLOCKSIZE>>>(dev_timeX, dev_magY, dev_magDY, dev_pgram, szData, minFreq, maxFreq, numFreqs);
		cudaFree(dev_magDY); 
	}else{

		tstart = omp_get_wtime();
		lombscargleOneObject_float<<< numBlocks, BLOCKSIZE>>>(dev_timeX, dev_magY, dev_pgram, szData, minFreq, maxFreq, numFreqs);
	}

  	

  	//Find the index of the maximum power
  	thrust::device_ptr<float> maxLoc;
  	thrust::device_ptr<float> dev_ptr_pgram = thrust::device_pointer_cast(dev_pgram);

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
  	*pgram=(float *)malloc(sizeof(float)*numFreqs);
  	//copy using pinned memory
  	float * pinned_buffer;
	unsigned int sizeBufferElems=(SIZEPINNEDBUFFERMIB*1024*1024)/(sizeof(float));
	gpuErrchk(cudaMallocHost((void**)&pinned_buffer, sizeof(float)*sizeBufferElems*NSTREAMS));	

  	unsigned int sizeDevData=numFreqs;
  	pinnedMemoryCopyDtoH_float(pinned_buffer, sizeBufferElems, dev_pgram, *pgram, sizeDevData);

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
void batchGPULS_float(unsigned int * objectId, float * timeX,  float * magY, float * magDY, unsigned int * sizeData, const float minFreq, const float maxFreq, const unsigned int numFreqs, float * sumPeriods, float ** pgram, float * foundPeriod)
{
	



	float * dev_timeX;
	float * dev_magY;
	unsigned int * dev_sizeData;
	float * dev_foundPeriod;
	float * dev_pgram;

	struct lookupObj * dev_objectLookup;

	//compute the object ranges in the arrays and store in struct
	//This is given by the objectId
	struct lookupObj * objectLookup=NULL;
	unsigned int numUniqueObjects;
	computeObjectRanges(objectId, sizeData, &objectLookup, &numUniqueObjects);

    foundPeriod=(float *)malloc(sizeof(float)*numUniqueObjects);

	//allocate memory on the GPU
	gpuErrchk(cudaMalloc((void**)&dev_timeX, sizeof(float)*(*sizeData)));
	gpuErrchk(cudaMalloc((void**)&dev_magY, sizeof(float)*(*sizeData)));

	//If astropy implementation with error
	if(errorMode){
		float * dev_magDY;
		gpuErrchk(cudaMalloc((void**)&dev_magDY, sizeof(float)*(*sizeData)));

		//Need to incorporate error into magnitudes
		for (int i=0; i<numUniqueObjects; i++)
		{
			unsigned int idxMin=objectLookup[i].idxMin;
			unsigned int idxMax=objectLookup[i].idxMax;
			unsigned int sizeDataForObject=idxMax-idxMin+1;
			updateYerrorfactor_float(&magY[idxMin], &magDY[idxMin], sizeDataForObject);
		}	
	}
	
	//Allocate pgram
	// Result periodogram must be number of unique objects * the size of the frequency array
	gpuErrchk(cudaMalloc((void**)&dev_pgram, sizeof(float)*numFreqs*numUniqueObjects));

	//Make a small pinned memory buffer for transferring the array back
	float * pinned_buffer;
	unsigned int sizeBufferElems=(SIZEPINNEDBUFFERMIB*1024*1024)/(sizeof(float));
	gpuErrchk(cudaMallocHost((void**)&pinned_buffer, sizeof(float)*sizeBufferElems*NSTREAMS));	

	//the maximum power in each periodogram. Use this when we don't want to return the periodogram
	gpuErrchk(cudaMalloc((void**)&dev_foundPeriod, sizeof(float)*numUniqueObjects));
	gpuErrchk(cudaMalloc((void**)&dev_sizeData, sizeof(unsigned int)));
	gpuErrchk(cudaMalloc((void**)&dev_objectLookup, sizeof(lookupObj)*(numUniqueObjects)));

	//copy data to the GPU
	gpuErrchk(cudaMemcpy( dev_timeX, timeX, sizeof(float)*(*sizeData), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy( dev_magY, magY, sizeof(float)*(*sizeData), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy( dev_sizeData, sizeData, sizeof(unsigned int), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy( dev_objectLookup, objectLookup, sizeof(lookupObj)*(numUniqueObjects), cudaMemcpyHostToDevice));


	const int numBlocks=numUniqueObjects;
	double tstart;
  	//Do lomb-scargle
  	if(errorMode){

		float * dev_magDY;
		gpuErrchk(cudaMalloc((void**)&dev_magDY, sizeof(float)*(*sizeData)));

		//Need to incorporate error into magnitudes
		for (int i=0; i<numUniqueObjects; i++)
		{
			unsigned int idxMin=objectLookup[i].idxMin;
			unsigned int idxMax=objectLookup[i].idxMax;
			unsigned int sizeDataForObject=idxMax-idxMin+1;
			updateYerrorfactor_float(&magY[idxMin], &magDY[idxMin], sizeDataForObject);
		}	

		gpuErrchk(cudaMemcpy( dev_magDY, magDY, sizeof(float)*(*sizeData), cudaMemcpyHostToDevice));

		tstart=omp_get_wtime();
		lombscargleBatchError_float<<< numBlocks, BLOCKSIZE>>>(dev_timeX, dev_magY, dev_magDY, dev_objectLookup, dev_pgram, dev_foundPeriod, minFreq, maxFreq, numFreqs);

		cudaFree(dev_magDY); 
	}else{
		tstart=omp_get_wtime();
		lombscargleBatch_float<<< numBlocks, BLOCKSIZE>>>(dev_timeX, dev_magY, dev_objectLookup, dev_pgram, dev_foundPeriod, minFreq, maxFreq, numFreqs);
	}

  	cudaDeviceSynchronize();

  	double tend=omp_get_wtime();
  	printf("\nTime to compute kernel: %f",tend-tstart);


  	

  	//copy pgram back to the host
  	*pgram=(float *)malloc(sizeof(float)*numFreqs*numUniqueObjects);
  	//copy pgram back using pinned memory
  	unsigned int sizeDevData=numUniqueObjects*numFreqs;
  	pinnedMemoryCopyDtoH_float(pinned_buffer, sizeBufferElems, dev_pgram, *pgram, sizeDevData);

  	//For each object, find the maximum power in the pgram
  	//then find the corresponding line in the periods file from daniel
  	
  	//compute the maximum power using the returned pgram
  	double tstartcpupgram=omp_get_wtime();
  	printf("\nCompute period from pgram on CPU:");

  	double freqStep=(maxFreq-minFreq)/(numFreqs*1.0);	
  	#pragma omp parallel for num_threads(4)
  	for (int i=0; i<numUniqueObjects; i++)
  	{
  		float maxPower=0;
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
  	

  	if(printMode){
		for (int i=0; i<numUniqueObjects; i++)
		{
			printf("\nObject: %d Period: %f, ",objectLookup[i].objId,foundPeriod[i]);
		}
	}


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


void importObjXYData_float(char * fnamedata, unsigned int * sizeData, unsigned int ** objectId, float ** timeX, float ** magY, float ** magDY)
{

	//import objectId, timeX, magY

	std::vector<float>tmpAllData;
	std::ifstream in(fnamedata);
	unsigned int cnt=0;
	for (std::string f; getline(in, f, ',');){

	float i;
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




  	if(errorMode){
		*sizeData=(unsigned int)tmpAllData.size()/4;
	}else{
		*sizeData=(unsigned int)tmpAllData.size()/3;
	}

  	printf("\nData import: Total rows: %u",*sizeData);
  	
  	*objectId=(unsigned int *)malloc(sizeof(float)*(*sizeData));
  	*timeX=   (float *)malloc(sizeof(float)*(*sizeData));
  	*magY=    (float *)malloc(sizeof(float)*(*sizeData));



	if(errorMode){
		*magDY=(float *)malloc(sizeof(float)*(*sizeData));
		for (int i=0; i<*sizeData; i++){
			(*objectId)[i]=tmpAllData[(i*4)+0];
			(*timeX)[i]   =tmpAllData[(i*4)+1];
			(*magY)[i]    =tmpAllData[(i*4)+2];
			(*magDY)[i]    =tmpAllData[(i*4)+3];
		}
	}else{
		for (int i=0; i<*sizeData; i++){
			(*objectId)[i]=tmpAllData[(i*3)+0];
			(*timeX)[i]   =tmpAllData[(i*3)+1];
			(*magY)[i]    =tmpAllData[(i*3)+2];
		}
	}

}

//Pre-center the data with error:		
// w = dy ** -2
//    y = y - np.dot(w, y) / np.sum(w)
void updateYerrorfactor_float(float * y, float *dy, const unsigned int sizeData)
{

		//Pre-center the data with error:
		//w = dy ** -2
		//sum w
		float * w =(float *)malloc(sizeof(float)*sizeData);
		float sumw=0;
		#pragma omp parallel for num_threads(4) reduction(+:sumw)
		for (int i=0; i<sizeData; i++)
		{
			w[i]=1.0/sqrt(dy[i]);
			sumw+=w[i];
		}
		//compute dot product w,y
		float dotwy=0;
		#pragma omp parallel for num_threads(4) reduction(+:dotwy)
		for (int i=0; i<sizeData; i++)
		{
			dotwy+=w[i]*y[i];
		}

		//update y to account for dot product and sum w
		//y = y - dot(w, y) / np.sum(w)	
		#pragma omp parallel for num_threads(4)
		for (int i=0; i<sizeData; i++)
		{
			y[i]=y[i]-dotwy/sumw;
		}

		free(w);
}

//Double Section

void pinnedMemoryCopyDtoH_double(double * pinned_buffer, unsigned int sizeBufferElems, double * dev_data, double * pageable, unsigned int sizeTotalData)
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
		gpuErrchk(cudaMemcpyAsync(pinned_buffer+pinnedBufferOffset, dev_data+(offsetstart), sizeof(double)*elemsToTransfer, cudaMemcpyDeviceToHost, streams[tid])); 

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
void GPULSOneObject_double(double * timeX,  double * magY, double * magDY, unsigned int * sizeData, const double minFreq, const double maxFreq, const unsigned int numFreqs, double * periodFound, double ** pgram)
{
	

	double * dev_timeX;
	double * dev_magY;
	unsigned int * dev_sizeData;
	double * dev_pgram;
	double * dev_foundPeriod;

	//allocate memory on the GPU
	gpuErrchk(cudaMalloc((void**)&dev_timeX, sizeof(double)*(*sizeData)));
	gpuErrchk(cudaMalloc((void**)&dev_magY, sizeof(double)*(*sizeData)));

	// Result periodogram
	//need to allocate it on the GPUeven if we do not return it to the host so that we can find the maximum power
	gpuErrchk(cudaMalloc((void**)&dev_pgram, sizeof(double)*numFreqs));
	
	//the maximum power in the periodogram. Use this when we don't want to return the periodogram
	gpuErrchk(cudaMalloc((void**)&dev_foundPeriod, sizeof(double)));
	gpuErrchk(cudaMalloc((void**)&dev_sizeData, sizeof(unsigned int)));

	//copy data to the GPU
	gpuErrchk(cudaMemcpy( dev_timeX, timeX, sizeof(double)*(*sizeData), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy( dev_magY, magY, sizeof(double)*(*sizeData), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy( dev_sizeData, sizeData, sizeof(unsigned int), cudaMemcpyHostToDevice));



	const unsigned int szData=*sizeData;
	const unsigned int numBlocks=ceil(numFreqs*1.0/BLOCKSIZE*1.0);
	
	double tstart;
	  //Do lomb-scargle
	if(errorMode){

		double * dev_magDY;
		gpuErrchk(cudaMalloc((void**)&dev_magDY, sizeof(double)*(*sizeData)));
		gpuErrchk(cudaMemcpy( dev_magDY, magDY, sizeof(double)*(*sizeData), cudaMemcpyHostToDevice));

		tstart=omp_get_wtime();
		lombscargleOneObjectError_double<<< numBlocks, BLOCKSIZE>>>(dev_timeX, dev_magY, dev_magDY, dev_pgram, szData, minFreq, maxFreq, numFreqs);
		cudaFree(dev_magDY); 

	}else{
		tstart=omp_get_wtime();
		lombscargleOneObject_double<<< numBlocks, BLOCKSIZE>>>(dev_timeX, dev_magY, dev_pgram, szData, minFreq, maxFreq, numFreqs);
	}
  	

  	//Find the index of the maximum power
  	thrust::device_ptr<double> maxLoc;
  	thrust::device_ptr<double> dev_ptr_pgram = thrust::device_pointer_cast(dev_pgram);

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
  	*pgram=(double *)malloc(sizeof(double)*numFreqs);
  	//copy using pinned memory
  	double * pinned_buffer;
	unsigned int sizeBufferElems=(SIZEPINNEDBUFFERMIB*1024*1024)/(sizeof(double));
	gpuErrchk(cudaMallocHost((void**)&pinned_buffer, sizeof(double)*sizeBufferElems*NSTREAMS));	

  	unsigned int sizeDevData=numFreqs;
  	pinnedMemoryCopyDtoH_double(pinned_buffer, sizeBufferElems, dev_pgram, *pgram, sizeDevData);

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
void batchGPULS_double(unsigned int * objectId, double * timeX,  double * magY, double * magDY, unsigned int * sizeData, const double minFreq, const double maxFreq, const unsigned int numFreqs, double * sumPeriods, double ** pgram, double * foundPeriod)
{
	



	double * dev_timeX;
	double * dev_magY;
	unsigned int * dev_sizeData;
	double * dev_foundPeriod;
	double * dev_pgram;

	struct lookupObj * dev_objectLookup;

	//compute the object ranges in the arrays and store in struct
	//This is given by the objectId
	struct lookupObj * objectLookup=NULL;
	unsigned int numUniqueObjects;
	computeObjectRanges(objectId, sizeData, &objectLookup, &numUniqueObjects);

    foundPeriod=(double *)malloc(sizeof(double)*numUniqueObjects);

	//allocate memory on the GPU
	gpuErrchk(cudaMalloc((void**)&dev_timeX, sizeof(double)*(*sizeData)));
	gpuErrchk(cudaMalloc((void**)&dev_magY, sizeof(double)*(*sizeData)));

	//If astropy implementation with error
	if(errorMode){
		double * dev_magDY;
		gpuErrchk(cudaMalloc((void**)&dev_magDY, sizeof(double)*(*sizeData)));
	
		//Need to incorporate error into magnitudes
		for (int i=0; i<numUniqueObjects; i++)
		{
			unsigned int idxMin=objectLookup[i].idxMin;
			unsigned int idxMax=objectLookup[i].idxMax;
			unsigned int sizeDataForObject=idxMax-idxMin+1;
			updateYerrorfactor_double(&magY[idxMin], &magDY[idxMin], sizeDataForObject);
		}	
	}
	
	
	//Allocate pgram
	// Result periodogram must be number of unique objects * the size of the frequency array
	gpuErrchk(cudaMalloc((void**)&dev_pgram, sizeof(double)*numFreqs*numUniqueObjects));

	//Make a small pinned memory buffer for transferring the array back
	double * pinned_buffer;
	unsigned int sizeBufferElems=(SIZEPINNEDBUFFERMIB*1024*1024)/(sizeof(double));
	gpuErrchk(cudaMallocHost((void**)&pinned_buffer, sizeof(double)*sizeBufferElems*NSTREAMS));	

	//the maximum power in each periodogram. Use this when we don't want to return the periodogram
	gpuErrchk(cudaMalloc((void**)&dev_foundPeriod, sizeof(double)*numUniqueObjects));
	gpuErrchk(cudaMalloc((void**)&dev_sizeData, sizeof(unsigned int)));
	gpuErrchk(cudaMalloc((void**)&dev_objectLookup, sizeof(lookupObj)*(numUniqueObjects)));

	//copy data to the GPU
	gpuErrchk(cudaMemcpy( dev_timeX, timeX, sizeof(double)*(*sizeData), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy( dev_magY, magY, sizeof(double)*(*sizeData), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy( dev_sizeData, sizeData, sizeof(unsigned int), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy( dev_objectLookup, objectLookup, sizeof(lookupObj)*(numUniqueObjects), cudaMemcpyHostToDevice));


	const int numBlocks=numUniqueObjects;
	double tstart;

  	//Do lomb-scargle
	if(errorMode){
		double * dev_magDY;
		gpuErrchk(cudaMalloc((void**)&dev_magDY, sizeof(double)*(*sizeData)));
	
		//Need to incorporate error into magnitudes
		for (int i=0; i<numUniqueObjects; i++)
		{
			unsigned int idxMin=objectLookup[i].idxMin;
			unsigned int idxMax=objectLookup[i].idxMax;
			unsigned int sizeDataForObject=idxMax-idxMin+1;
			updateYerrorfactor_double(&magY[idxMin], &magDY[idxMin], sizeDataForObject);
		}	
		gpuErrchk(cudaMemcpy( dev_magDY, magDY, sizeof(double)*(*sizeData), cudaMemcpyHostToDevice));

		tstart=omp_get_wtime();
		lombscargleBatchError_double<<< numBlocks, BLOCKSIZE>>>(dev_timeX, dev_magY, dev_magDY, dev_objectLookup, dev_pgram, dev_foundPeriod, minFreq, maxFreq, numFreqs);
		cudaFree(dev_magDY); 
	}else{
		tstart=omp_get_wtime();
		lombscargleBatch_double<<< numBlocks, BLOCKSIZE>>>(dev_timeX, dev_magY, dev_objectLookup, dev_pgram, dev_foundPeriod, minFreq, maxFreq, numFreqs);
	}

  	cudaDeviceSynchronize();

  	double tend=omp_get_wtime();
  	printf("\nTime to compute kernel: %f",tend-tstart);


  	

  	//copy pgram back to the host
  	*pgram=(double *)malloc(sizeof(double)*numFreqs*numUniqueObjects);
  	//copy pgram back using pinned memory
  	unsigned int sizeDevData=numUniqueObjects*numFreqs;
  	pinnedMemoryCopyDtoH_double(pinned_buffer, sizeBufferElems, dev_pgram, *pgram, sizeDevData);

  	//For each object, find the maximum power in the pgram
  	//then find the corresponding line in the periods file from daniel
  	
  	//compute the maximum power using the returned pgram
  	double tstartcpupgram=omp_get_wtime();
  	printf("\nCompute period from pgram on CPU:");

  	double freqStep=(maxFreq-minFreq)/(numFreqs*1.0);	
  	#pragma omp parallel for num_threads(4)
  	for (int i=0; i<numUniqueObjects; i++)
  	{
  		double maxPower=0;
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
  	

  	if(printMode){
		for (int i=0; i<numUniqueObjects; i++)
		{
			printf("\nObject: %d Period: %f, ",objectLookup[i].objId,foundPeriod[i]);
		}
	}


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


void importObjXYData_double(char * fnamedata, unsigned int * sizeData, unsigned int ** objectId, double ** timeX, double ** magY, double ** magDY)
{

	//import objectId, timeX, magY

	std::vector<double>tmpAllData;
	std::ifstream in(fnamedata);
	unsigned int cnt=0;
	for (std::string f; getline(in, f, ',');){

	double i;
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



	if(errorMode){
	*sizeData=(unsigned int)tmpAllData.size()/4;
	}else{
	*sizeData=(unsigned int)tmpAllData.size()/3;
	}

  	printf("\nData import: Total rows: %u",*sizeData);
  	
  	*objectId=(unsigned int *)malloc(sizeof(double)*(*sizeData));
  	*timeX=   (double *)malloc(sizeof(double)*(*sizeData));
  	*magY=    (double *)malloc(sizeof(double)*(*sizeData));


	if(errorMode){

		*magDY=(double *)malloc(sizeof(double)*(*sizeData));

		for (int i=0; i<*sizeData; i++){
			(*objectId)[i]=tmpAllData[(i*4)+0];
			(*timeX)[i]   =tmpAllData[(i*4)+1];
			(*magY)[i]    =tmpAllData[(i*4)+2];
			(*magDY)[i]    =tmpAllData[(i*4)+3];
		}
	}else{

		for (int i=0; i<*sizeData; i++){
			(*objectId)[i]=tmpAllData[(i*3)+0];
			(*timeX)[i]   =tmpAllData[(i*3)+1];
			(*magY)[i]    =tmpAllData[(i*3)+2];
		}
	}


}

//Pre-center the data with error:		
// w = dy ** -2
//    y = y - np.dot(w, y) / np.sum(w)
void updateYerrorfactor_double(double * y, double *dy, const unsigned int sizeData)
{

		//Pre-center the data with error:
		//w = dy ** -2
		//sum w
		double * w =(double *)malloc(sizeof(double)*sizeData);
		double sumw=0;
		#pragma omp parallel for num_threads(4) reduction(+:sumw)
		for (int i=0; i<sizeData; i++)
		{
			w[i]=1.0/sqrt(dy[i]);
			sumw+=w[i];
		}
		//compute dot product w,y
		double dotwy=0;
		#pragma omp parallel for num_threads(4) reduction(+:dotwy)
		for (int i=0; i<sizeData; i++)
		{
			dotwy+=w[i]*y[i];
		}

		//update y to account for dot product and sum w
		//y = y - dot(w, y) / np.sum(w)	
		#pragma omp parallel for num_threads(4)
		for (int i=0; i<sizeData; i++)
		{
			y[i]=y[i]-dotwy/sumw;
		}

		free(w);
}

