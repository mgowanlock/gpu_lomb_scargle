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
template <typename DTYPE>
void importObjXYData(char * fnamedata, unsigned int * sizeData, unsigned int ** objectId, DTYPE ** timeX, DTYPE ** magY, DTYPE ** magDY);

//CPU L-S Functions:
template <typename DTYPE>
void lombscarglecpu(bool mode, DTYPE * x, DTYPE * y, const unsigned int sizeData, const unsigned int numFreqs, const DTYPE minFreq, const DTYPE maxFreq, const DTYPE freqStep, DTYPE * pgram);
template <typename DTYPE>
void lombscarglecpuinnerloop(int iteration, DTYPE * x, DTYPE * y, DTYPE * pgram, DTYPE * freqToTest, const unsigned int sizeData);
template <typename DTYPE>
void lombscargleCPUOneObject(DTYPE * timeX,  DTYPE * magY, unsigned int * sizeData, const DTYPE minFreq, const DTYPE maxFreq, const unsigned int numFreqs, DTYPE * foundPeriod, DTYPE * foundPower, DTYPE * pgram);
template <typename DTYPE>
void lombscargleCPUBatch(unsigned int * objectId, DTYPE * timeX,  DTYPE * magY, unsigned int * sizeData, const DTYPE minFreq, const DTYPE maxFreq, const unsigned int numFreqs, DTYPE * sumPeriods, DTYPE * pgram, DTYPE * foundPeriod, DTYPE * foundPower);
template <typename DTYPE>
void normalizepgram(DTYPE * pgram, struct lookupObj * objectLookup, DTYPE * magY, uint64_t numUniqueObjects, uint64_t numFreqs);
template <typename DTYPE>
void normalizepgramsingleobject(DTYPE * pgram, DTYPE * magY, uint64_t sizeDataForObject, uint64_t numFreqs);
// double computeMeanDataSquared(DTYPE * magY, uint64_t sizeData);
template <typename DTYPE>
double computeStandardDevSquared(DTYPE * magY, uint64_t sizeData);

//With error
template <typename DTYPE>
void lombscargleCPUOneObjectError(DTYPE * time, DTYPE * magY, DTYPE * magDY, unsigned int * sizeData, const DTYPE minFreq, const DTYPE maxFreq, const unsigned int numFreqs, DTYPE * foundPeriod, DTYPE * foundPower, DTYPE * pgram);
template <typename DTYPE>
void lombscarglecpuError(bool mode, DTYPE * x, DTYPE * y, DTYPE *dy, const unsigned int sizeData, const unsigned int numFreqs, const DTYPE minFreq, const DTYPE maxFreq, const DTYPE freqStep, DTYPE * pgram);
template <typename DTYPE>
void lombscarglecpuinnerloopAstroPy(int iteration, DTYPE * x, DTYPE * y, DTYPE * dy, DTYPE * pgram, DTYPE * freqToTest, const unsigned int sizeData);
template <typename DTYPE> 
void lombscargleCPUBatchError(unsigned int * objectId, DTYPE * timeX,  DTYPE * magY, DTYPE * magDY, unsigned int * sizeData, const DTYPE minFreq, const DTYPE maxFreq, const unsigned int numFreqs, DTYPE * sumPeriods, DTYPE * pgram, DTYPE * foundPeriod, DTYPE * foundPower);
template <typename DTYPE>
void updateYerrorfactor(DTYPE * y, DTYPE *dy, const unsigned int sizeData);

//GPU functions
template <typename DTYPE>
void batchGPULS(unsigned int * objectId, DTYPE * timeX,  DTYPE * magY, DTYPE * magDY, unsigned int * sizeData, const DTYPE minFreq, const DTYPE maxFreq, const unsigned int numFreqs, DTYPE * sumPeriods, DTYPE * pgram, DTYPE * foundPeriod, DTYPE * foundPower);
template <typename DTYPE>
void GPULSOneObject(DTYPE * timeX,  DTYPE * magY, DTYPE * magDY, unsigned int * sizeData, const DTYPE minFreq, const DTYPE maxFreq, const unsigned int numFreqs, DTYPE * periodFound, DTYPE * maxPowerFound, DTYPE ** pgram);
void computeObjectRanges(unsigned int * objectId, unsigned int * sizeData, struct lookupObj ** objectLookup, unsigned int * numUniqueObjects);
template <typename DTYPE>
void pinnedMemoryCopyDtoH(DTYPE * pinned_buffer, unsigned int sizeBufferElems, DTYPE * dev_data, DTYPE * pageable, unsigned int sizeTotalData);
template <typename DTYPE>
void computePeriod(DTYPE * pgram, const unsigned int numFreqs, const DTYPE minFreq, const DTYPE freqStep, DTYPE * foundPeriod, DTYPE * foundPower);
template <typename DTYPE>
unsigned int computeNumBatches(bool mode, bool pgrammode, unsigned int totalLengthTimeSeries, unsigned int numObjects, unsigned int numFreq, DTYPE dataType);
double getGPUCapacity();
void warmUpGPU();

//for Batching and multi-GPU for batch mode
template <typename DTYPE>
void batchGPULSWrapper(unsigned int * objectId, DTYPE * timeX,  DTYPE * magY, DTYPE * magDY, unsigned int * sizeData, const DTYPE minFreq, const DTYPE maxFreq, const unsigned int numFreqs, DTYPE * sumPeriods, DTYPE ** pgram, DTYPE ** foundPeriod, DTYPE ** foundPower);

//output to files and stdout:
template <typename DTYPE>
void outputPeriodsToFile(struct lookupObj * objectLookup, unsigned int numUniqueObjects, DTYPE * foundPeriod, DTYPE * foundPower);
template <typename DTYPE>
void outputPeriodsToFileTopThree(struct lookupObj * objectLookup, unsigned int numUniqueObjects, DTYPE * foundPeriod, DTYPE * foundPower);
template <typename DTYPE>
void outputPgramToFile(struct lookupObj * objectLookup, unsigned int numUniqueObjects, unsigned int numFreqs, DTYPE ** pgram);
template <typename DTYPE>
void outputPeriodsToStdout(struct lookupObj * objectLookup, unsigned int numUniqueObjects, DTYPE * foundPeriod, DTYPE * foundPower);

template <typename DTYPE>
void executer(char * inputFname, DTYPE minFreq, DTYPE maxFreq, const unsigned int freqToTest, DTYPE dataType);

using namespace std;

// runtime parameters set to default values
int errorMode = 0;
int dMode = 0;
int nThreadsCpu = 4;
int numGPU = 1;
int returnPgram = 1;
int printPeriods = 2;
int printPgram = 0;
int normalizePgram = 0;
int computeMode;
char periodOutFileName[500];
char pgramOutFileName[500];
bool quiet = false;

int nStreams = 3;


//parrsing functions for command line arguments
char* getCmdOption(char ** begin, char ** end, const std::string & option1, const std::string & option2)
{

	if(std::find(begin, end, option1) != end){
		char ** itr = std::find(begin, end, option1);
		if (itr != end && ++itr != end)
		{
			return *itr;
		}

	} else if(std::find(begin, end, option1) != end){
		char ** itr = std::find(begin, end, option2);
		if (itr != end && ++itr != end)
		{
			return *itr;
		}

	}

	return 0;
}

bool cmdOptionExists(char** begin, char** end, const std::string& option)
{
    return std::find(begin, end, option) != end;
}

int main(int argc, char *argv[])
{


	/////////////////////////
	// Get information from command line
	/////////////////////////
	double minFreq, maxFreq;
	int freqToTest;


	
	if(cmdOptionExists(argv, argv+argc, "-h") || cmdOptionExists(argv, argv+argc, "--help") ){
        // print out all options etc
		printf("This code computes the period/s of object/s. The rutime arguments are below:\n\n");
		printf("-q, --quiet             This option supresses the messages that give guidence on\n");
		printf("                        how to use runtime comands for this software.\n\n");						
		printf("-f, --filename <data>   This flag is required to specify the path to the data.\n\n");
		printf("-e, --error             This flag identifies if there are error amounts incldued\n");
		printf("                        in the observations in the data file.\n\n");
		printf("-m,--mode               This flag is required and checks if there are multiple\n");
		printf("                        objects in the file, and what device the LS will run on.\n");
		printf("                        The flag codes are as follows:\n");
		printf("                        Please enter a compute mode with -m <1-4> or --mode <1-4>:\n");
		printf("                             1: Multiple Objects with a GPU\n");
		printf("                             2: Single Object with a GPU\n");
		printf("                             3: Multiple Objects on the CPU\n");
		printf("                             4: Single Object on the CPU\n\n");
		printf("-d, --double            This flag is used to select a double datatype to allow for\n");
		printf("                        more accurate compuation at the cost of performance.\n\n");
		printf("-ps, --printscreen      This flag causes the periods of all objects to be printed\n");
		printf("                        to the screen during runtime.\n\n");
		printf("-pf <file>,             This option prints the periods to a file at <file> instead\n");
		printf("  --printfile < file>   of printing them to the screen.\n\n");
		printf("-min, --minfreq         This required flag sets the minimum frequency to test.\n\n");
		printf("-max, --maxfreq         This required flag sets the maximum frequency to test.\n\n");
		printf("-fq --freq              This required flag sets the number of frequencies to test.\n\n");
		printf("-n, -nthreads           This option can be used to set the number of cpu threads.\n\n");
		printf("-ng, --ngpus            This option can be sued to specify the number of avaliable\n");
		printf("                        GPUs. It defaults to 1.\n\n");
		printf("-ppf <file>,            This option will print the periodogram to the file location.\n");
		printf("--printpgramfile <file>\n\n");
		printf("-nrp, --normpgram       This option will normalize the returned periodogram.\n\n");

		printf("NOTE: Do not forget to change the compute cabability for the GPU generation at\n");
		printf("compile time. Failure to do so will cause the program to crash.\n\n");


		return 0;
    }

	if(cmdOptionExists(argv, argv+argc, "-q") || cmdOptionExists(argv, argv+argc, "--quiet") ){
        // quiet mode
		quiet = true;
    } else {
		printf("To enable quiet mode and suppress a majority of messages use -q or --quiet");
	}

	char inputFname[500];
	if(cmdOptionExists(argv, argv+argc, "-f") || cmdOptionExists(argv, argv+argc, "--filename")){
		//get file name
		strcpy(inputFname, getCmdOption(argv, argv + argc, "-f", "--filename"));
		if(!quiet) printf("\nDataset file: %s\n",inputFname);
	} else {
		printf("\nA filename is required. Use -f or --filenmae to specify the path to a file.\n");
		return 0;
	}

	if(cmdOptionExists(argv, argv+argc, "-e") || cmdOptionExists(argv, argv+argc, "--error")){
		//set the error mode
		errorMode = 1;
		if(!quiet) printf("Error mode has been set to on. (The option to return the Pgram will be set to on as well.)\n");
		returnPgram = 1;
	} else {
		if(!quiet) printf("Error mode has defaulted to off. To enable error mode please use -e or --error\n");
	}

	if(cmdOptionExists(argv, argv+argc, "-m") || cmdOptionExists(argv, argv+argc, "--mode")){
		//set the computation mode
		computeMode = atoi(getCmdOption(argv, argv + argc, "-m", "--mode"));
		if(computeMode ==1){
			if(!quiet) printf("Compute mode set for multiple objects with a GPU.\n");
		}else if(computeMode == 2){
			if(!quiet) printf("Compute mode set for a single object with a GPU.\n");
		}else if(computeMode == 3){
			if(!quiet) printf("Compute mode set for multiple objects on a CPU.\n");
		}else if(computeMode == 4){
			if(!quiet) printf("Compute mode set for a single object on the CPU.\n");
		}

	} else {
		printf("Please enter a compute mode with -m <1-4> or --mode <1-4>:\n     1: Multiple Objects with a GPU\n     2: Single Object with a GPU\n     3: Multiple Objects on the CPU\n     4: Single Object on the CPU\n");
		return 0;
	}

	if(cmdOptionExists(argv, argv+argc, "-d") || cmdOptionExists(argv, argv+argc, "--double")){
		//set the datatype
		dMode = 1;
		if(!quiet) printf("Data type has been set to 64-bit doubles.\n");

	}else{
		if(!quiet) printf("Data type has defaulted to 32-bit floating point numbers. If additional accuracy is required use the -d or --double option for 64-bit precision.\n");
	}

	if(computeMode == 1 || computeMode == 3){
		if(cmdOptionExists(argv, argv+argc, "-ps") || cmdOptionExists(argv, argv+argc, "--printscreen")){
			//set the print mode
			printPeriods = 1;
			if(!quiet) printf("Printing the period/s of the object/s to screen.\n");
		}else if(cmdOptionExists(argv, argv+argc, "-pf") || cmdOptionExists(argv, argv+argc, "--printfile")){
			printPeriods = 2;
			periodOutFileName[0] = 0;
			strcpy(periodOutFileName, getCmdOption(argv, argv + argc, "-pf", "--printfile"));
			if(periodOutFileName[0] == 0 || periodOutFileName[0] == '-'){
				printf("Please Enter a valid output filename to print periods.\n");
				return 0;
			}
			if(!quiet) printf("The period/s will be printed to the file %s\n",periodOutFileName);
		}else {
			if(!quiet) printf("By defualt the periods of the object/s will not be printed. To enable print to screen use -ps or --printscreen, to print to file use -pf <filename> or --printfile <filename>\n");
		}
	}


	if(cmdOptionExists(argv, argv+argc, "-min") || cmdOptionExists(argv, argv+argc, "--minfreq")){
		//set min frequency
		minFreq = atof(getCmdOption(argv, argv + argc, "-min", "--minfreq"));
		if(!quiet) printf("Minimum Frequency has ben set to %f.\n", minFreq);
	} else {
		printf("Please use -min <freq> or -- minfreq <freq> to specify a minimum freuency.\n");
		return 0;
	}

	if(cmdOptionExists(argv, argv+argc, "-max") || cmdOptionExists(argv, argv+argc, "--maxfreq")){
		//set max frequency
		maxFreq = atof(getCmdOption(argv, argv + argc, "-max", "--maxfreq"));
		if(!quiet) printf("Maximum frequency has ben set to %f.\n", maxFreq);
	} else {
		printf("Please use -max <freq> or -- maxfreq <freq> to specify a maximum freuency.\n");
		return 0;
	}

	if(cmdOptionExists(argv, argv+argc, "-fq") || cmdOptionExists(argv, argv+argc, "--freq")){
		//set number of frequencies to test
		freqToTest = atoi(getCmdOption(argv, argv + argc, "-fq", "--freq"));
		if(!quiet) printf("The number of frequncies to test has been set to %d.\n", freqToTest);
	}else{
		printf("Please enter the number of frequencies to test with -fq <value> or --freq <value>\n");
		return 0;
	}

	if(cmdOptionExists(argv, argv+argc, "-n") || cmdOptionExists(argv, argv+argc, "--nthreads")){
		//set number of cpu threads
		nThreadsCpu = atoi(getCmdOption(argv, argv + argc, "-n", "--nthreads"));
		if(!quiet) printf("Number of CPU threads set to %d.\n", nThreadsCpu);
	}else{
		if(!quiet) printf("Default number of CPU threads set to %d\n", nThreadsCpu);
	}

	if(computeMode == 1 || computeMode == 2){
		if(cmdOptionExists(argv, argv+argc, "-ng") || cmdOptionExists(argv, argv+argc, "--ngpus")){
			//set number of gpus to use
			numGPU = atoi(getCmdOption(argv, argv + argc, "-ng", "--ngpus"));
		if(!quiet) printf("Number of GPUS set to %d.\n",numGPU);
		} else {
			if(!quiet) printf("Default number of GPUs set to 1. To specify a different number of GPUs use -ng <value> or --ngpus <value>\n");
		}
	}

	// only return the pg if printing or saving to file
	// if(cmdOptionExists(argv, argv+argc, "-pg") || cmdOptionExists(argv, argv+argc, "--pgram")){
	// 	//set return of pgram
	// }


	// if(cmdOptionExists(argv, argv+argc, "-pps") || cmdOptionExists(argv, argv+argc, "--printpgramscreen")){
	// 	//set the print mode
	// 	printPgram = 2;
	// 	returnPgram = 1;
	// 	if(!quiet) printf("Printing the pgram/s of the object/s to screen.\n");
	// }else
	if(cmdOptionExists(argv, argv+argc, "-ppf") || cmdOptionExists(argv, argv+argc, "--printpgramfile")){
		printPgram = 1;
		returnPgram = 1;
		pgramOutFileName[0] = 0;
		strcpy(pgramOutFileName, getCmdOption(argv, argv + argc, "-ppf", "--printpgramfile"));
		if(pgramOutFileName[0] == 0 || pgramOutFileName[0] == '-'){
			printf("Please Enter a valid output filename to print pgrams.\n");
			return 0;
		}
		if(!quiet) printf("The pgram/s will be printed to the file %s\n",periodOutFileName);
	}else {
		if(!quiet) printf("By defualt the pgrams of the object/s will not be printed. To enableprint to file use -ppf <filename> or --printpgramfile <filename>\n");
	}

	if(printPgram == 1){
		if(returnPgram == 1){
			if(cmdOptionExists(argv, argv+argc, "-nrp") || cmdOptionExists(argv, argv+argc, "--normpgram")){
				//set normalization of pgram
				normalizePgram = 1;
				if(!quiet) printf("The pgram will be normalized.\n");
			} else {
				if(!quiet) printf("By default the pgram will not be normalized. To normalize it use option -nrp or --normpgram.\n");
			}
		}
	}


	
	if(dMode == 1){
		double dataType = 1.0;
		executer(inputFname, minFreq, maxFreq, freqToTest, dataType);
	} else {
		float dataType = 1.0;
		executer(inputFname, (float)minFreq, (float)maxFreq, freqToTest, dataType);
	}

	printf("\n");
	return 0;
}


template <typename DTYPE>
void executer(char * inputFname, DTYPE minFreq, DTYPE maxFreq, const unsigned int freqToTest, DTYPE dataType){
	
	warmUpGPU();
	cudaProfilerStart();
	omp_set_nested(1);


	//validation and output to file
	char fname[]="gpu_stats.txt";
	ofstream gpu_stats;
	gpu_stats.open(fname,ios::app);	


	/////////////
	//Import Data
	/////////////
	unsigned int * objectId=NULL; 
	DTYPE * timeX=NULL; 
	DTYPE * magY=NULL;
	DTYPE * magDY=NULL;
	unsigned int sizeData;
	importObjXYData(inputFname, &sizeData, &objectId, &timeX, &magY, &magDY);

	// for (int i=0; i<10000; i++)
	// {
	// 	printf("\nobjectId: %d, %f, %f, %f", objectId[i], timeX[i],magY[i], magDY[i]);	
	// }
	// return 0;
	
	//pgram allocated in the functions below
	//Stores the LS power for each frequency
	DTYPE * pgram=NULL;

	//foundPeriod is allocated in the batch functions below
	DTYPE * foundPeriod=NULL;
	//foundPower is allocated in the batch functions below
	DTYPE * foundPower=NULL;

	//Batch of LS to compute on the GPU
	if (computeMode==1)
	{
		DTYPE sumPeriods=0;
		
		double tstart=omp_get_wtime();
		
		//original before using multiple GPUs and batching
		// batchGPULS(objectId, timeX, magY, magDY, &sizeData, minFreq, maxFreq, freqToTest, &sumPeriods, &pgram, foundPeriod);
		batchGPULSWrapper(objectId, timeX, magY, magDY, &sizeData, minFreq, maxFreq, freqToTest, &sumPeriods, &pgram, &foundPeriod, &foundPower);
		
		double tend=omp_get_wtime();
		double totalTime=tend-tstart;
		printf("\nTotal time to compute batch: %f", totalTime);
		printf("\n[Validation] Sum of all periods: %f", sumPeriods);

		gpu_stats<<totalTime<<", "<< inputFname<<", Sum of periods: "<<sumPeriods<<", Min/Max Freq: "<<minFreq<<"/"<<maxFreq<<",  Num tested freq: "<<freqToTest<<", MODE: "<<computeMode<<", NTHREADSCPU/NUMGPU/BLOCKSIZE/ERROR/RETURNPGRAM/nStreams/DTYPE: "<<nThreadsCpu<<", "<<numGPU<<", "<<BLOCKSIZE<<", "<<errorMode<<", "<<returnPgram<<", "<<nStreams<<", "<<STR(DTYPE)<<endl;
	}
	//One object to compute on the GPU
	else if (computeMode==2)
	{
		DTYPE periodFound=0;	
		DTYPE maxPowerFound=0;	
		double tstart=omp_get_wtime();
		
		if(errorMode == 1){
			updateYerrorfactor(magY, magDY, sizeData);
		}


		GPULSOneObject(timeX, magY, magDY, &sizeData, minFreq, maxFreq, freqToTest, &periodFound, &maxPowerFound, &pgram);
		
		double tend=omp_get_wtime();
		double totalTime=tend-tstart;
		printf("\nTotal time to compute batch: %f", totalTime);
		printf("\n[Validation] Period: %f, Power: %f", periodFound, maxPowerFound);

		gpu_stats<<totalTime<<", "<< inputFname<<", Sum of periods: "<<periodFound<<", Min/Max Freq: "<<minFreq<<"/"<<maxFreq<<",  Num tested freq: "<<freqToTest<<", MODE: "<<computeMode<<", NTHREADSCPU/BLOCKSIZE/ERROR/RETURNPGRAM/nStreams/DTYPE: "<<nThreadsCpu<<", "<<BLOCKSIZE<<", "<<errorMode<<", "<<returnPgram<<", "<<nStreams<<", "<<STR(DTYPE)<<endl;
	}
	//CPU- batch processing
	else if (computeMode==3)
	{
		DTYPE sumPeriods=0;
		double tstart=omp_get_wtime();
		if(errorMode == 1){
			lombscargleCPUBatchError(objectId, timeX, magY, magDY, &sizeData, minFreq, maxFreq, freqToTest, &sumPeriods, pgram, foundPeriod, foundPower);

		}else{
			lombscargleCPUBatch(objectId, timeX, magY, &sizeData, minFreq, maxFreq, freqToTest, &sumPeriods, pgram, foundPeriod, foundPower);
		}

		double tend=omp_get_wtime();
		double totalTime=tend-tstart;
		printf("\nTotal time to compute batch: %f", totalTime);
		printf("\n[Validation] Sum of all periods: %f", sumPeriods);
		gpu_stats<<totalTime<<", "<< inputFname<<", Sum of periods: "<<sumPeriods<<", Min/Max Freq: "<<minFreq<<"/"<<maxFreq<<",  Num tested freq: "<<freqToTest<<", MODE: "<<computeMode<<", NTHREADSCPU: "<<nThreadsCpu<<", ERROR: "<<errorMode<<", DTYPE: "<<STR(DTYPE)<<endl;
	}
	//CPU- one object
	else if (computeMode==4)
	{
		DTYPE foundPeriod=0;
		DTYPE foundPower=0;
		double tstart=omp_get_wtime();
		if(errorMode == 1){
			lombscargleCPUOneObjectError(timeX, magY, magDY, &sizeData, minFreq, maxFreq, freqToTest, &foundPeriod, &foundPower, pgram);
		}else{
			lombscargleCPUOneObject(timeX, magY, &sizeData, minFreq, maxFreq, freqToTest, &foundPeriod, &foundPower, pgram);
		}

		double tend=omp_get_wtime();
		double totalTime=tend-tstart;
		printf("\nTotal time to compute pgram (one object): %f", totalTime);
		printf("\n[Validation] Period: %f, Power: %f", foundPeriod, foundPower);
		gpu_stats<<totalTime<<", "<< inputFname<<", Sum of periods: "<<foundPeriod<<", Min/Max Freq: "<<minFreq<<"/"<<maxFreq<<",  Num tested freq: "<<freqToTest<<", MODE: "<<computeMode<<", NTHREADSCPU: "<<nThreadsCpu<<", ERROR: "<<errorMode<<", DTYPE: "<<STR(DTYPE)<<endl;
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

}


//Estimated memory footprint used to compute the number of batches
//used to compute the number of batches
//mode-0 is original
//mode-1 is floating mean, error propogation
//pgrammode-0 do not return the pgram
//pgrammode-1 return the pgram
//numObjects-- number of objects in the file
//totalLengthTimeSeries-- total lines in the file across all objects
//numFreq- number of frequencies searched
//pass in the underestimated capacity 
template <typename DTYPE>
unsigned int computeNumBatches(bool mode, bool pgrammode, unsigned int totalLengthTimeSeries, unsigned int numObjects, unsigned int numFreq, DTYPE dataType)
{

  //L-S space complexity (original)
  //2Nt+NoNf //Nt-length of time series, No-number of objects, Nf-number of frequencies searched		

  //L-S space complexity (with error)
  //3Nt+NoNf //Nt-length of time series, No-number of objects, Nf-number of frequencies searched		

  printf("\n*********************");
  double underestGPUcapacityGiB=getGPUCapacity();
  
  double totalGiB=0;

  //original L-S (no error)
  if (mode==0)
  {
  totalGiB+=(sizeof(DTYPE)*(2.0*totalLengthTimeSeries))/(1024*1024*1024.0);
  }
  //L-S (with error)
  if (mode==1)
  {
  totalGiB+=(sizeof(DTYPE)*(3.0*totalLengthTimeSeries))/(1024*1024*1024.0);	
  }

  //pgram
  if (pgrammode==1)
  {
  double pgramsize=(sizeof(DTYPE)*(1.0*numObjects*numFreq))/(1024*1024*1024.0);		
  totalGiB+=pgramsize;		
  printf("\nSize of pgram: %f (GiB)", pgramsize);
  }

  
  printf("\nEstimated global memory footprint (GiB): %0.9f", totalGiB);

  unsigned int numBatches=ceil(totalGiB/(underestGPUcapacityGiB));
  printf("\nMinimum number of batches: %u", numBatches);
  numBatches=ceil((numBatches*1.0/numGPU))*numGPU;
  printf("\nNumber of batches (after ensuring batches evenly divide %d GPUs): %u", numGPU, numBatches);

  printf("\n*********************\n");
  return numBatches;
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
	printf("\nNumber of unique objects: %u",*numUniqueObjects);



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

template <typename DTYPE>
void pinnedMemoryCopyDtoH(DTYPE * pinned_buffer, unsigned int sizeBufferElems, DTYPE * dev_data, DTYPE * pageable, unsigned int sizeTotalData)
{

  	cudaStream_t streams[nStreams];
  	//create stream for the device
	for (int i=0; i<nStreams; i++)
	{
	cudaStreamCreate(&streams[i]);
	}
  	

  	unsigned int numIters=sizeTotalData/sizeBufferElems;

  	unsigned int totalelemstransfered=0;
  	#pragma omp parallel for num_threads(nStreams) reduction(+:totalelemstransfered)
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

	for (int i=0; i<nStreams; i++)
	{
	cudaStreamDestroy(streams[i]);
	}
}









//Compute pgram for one object, not a batch of objects
template <typename DTYPE>
void GPULSOneObject(DTYPE * timeX,  DTYPE * magY, DTYPE * magDY, unsigned int * sizeData, const DTYPE minFreq, const DTYPE maxFreq, const unsigned int numFreqs, DTYPE * periodFound, DTYPE * maxPowerFound, DTYPE ** pgram)
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
	DTYPE * dev_magDY;
	if(errorMode ==1){
		gpuErrchk(cudaMalloc((void**)&dev_magDY, sizeof(DTYPE)*(*sizeData)));	
	}

	
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

	if(errorMode ==1){
		gpuErrchk(cudaMemcpy( dev_magDY, magDY, sizeof(DTYPE)*(*sizeData), cudaMemcpyHostToDevice));
	}

	const unsigned int szData=*sizeData;
	const unsigned int numBlocks=ceil(numFreqs*1.0/BLOCKSIZE*1.0);
	
	double tstart=omp_get_wtime();
  	//Do lomb-scargle
	if(errorMode ==1){
		lombscargleOneObjectError<DTYPE><<< numBlocks, BLOCKSIZE>>>(dev_timeX, dev_magY, dev_magDY, dev_pgram, szData, minFreq, maxFreq, numFreqs);
	}else{
		lombscargleOneObject<DTYPE><<< numBlocks, BLOCKSIZE>>>(dev_timeX, dev_magY, dev_pgram, szData, minFreq, maxFreq, numFreqs);

	}



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
	DTYPE * pinned_buffer;
  	if(returnPgram == 1){
		*pgram=(DTYPE *)malloc(sizeof(DTYPE)*numFreqs);

		unsigned int sizeBufferElems=(8*1024*1024)/(sizeof(DTYPE));
		gpuErrchk(cudaMallocHost((void**)&pinned_buffer, sizeof(DTYPE)*sizeBufferElems*nStreams));	

		unsigned int sizeDevData=numFreqs;
		pinnedMemoryCopyDtoH(pinned_buffer, sizeBufferElems, dev_pgram, *pgram, sizeDevData);

		if(normalizePgram == 1 && errorMode ==0){
			normalizepgramsingleobject(*pgram, magY, *sizeData, numFreqs);
		}

		*maxPowerFound=(*pgram)[maxPowerIdx];
		printf("\nMaximum power at found period: %f", *maxPowerFound);

		// fprintf(stderr,"Total elements transferred: %u",totalelemstransfered);

	}



  	///////////////////////
  	//Output
  
  	//Output pgram to file
	if(printPgram ==1){
		printf("\nPrinting the pgram to file: %s", pgramOutFileName);
		ofstream pgramoutput;
		pgramoutput.open(pgramOutFileName,ios::out);	
		pgramoutput.precision(4);
		for (int i=0; i<numFreqs; i++)
		{
			pgramoutput<<(*pgram)[i]<<", ";
		}
		pgramoutput<<endl;
		pgramoutput.close();
  
	}

  	//End output
  	///////////////////////




  	//free memory-- CUDA

  	cudaFree(dev_timeX);
  	cudaFree(dev_magY);
	cudaFree(dev_sizeData);
	cudaFree(dev_pgram);
	cudaFree(dev_foundPeriod);

	if(errorMode == 1){
		cudaFree(dev_magDY);
	}

	if(returnPgram == 1){
		cudaFreeHost(pinned_buffer);
	}
	
	
  	


}




















//Wrapper around main function for multi-GPU and batching
template <typename DTYPE>
void batchGPULSWrapper(unsigned int * objectId, DTYPE * timeX,  DTYPE * magY, DTYPE * magDY, unsigned int * sizeData, const DTYPE minFreq, const DTYPE maxFreq, const unsigned int numFreqs, DTYPE * sumPeriods, DTYPE ** pgram, DTYPE ** foundPeriod, DTYPE ** foundPower)
{
	//compute total number of unqiue objects in the file and their sizes
	struct lookupObj * objectLookup=NULL;
	unsigned int numUniqueObjects;
	computeObjectRanges(objectId, sizeData, &objectLookup, &numUniqueObjects);
	DTYPE dataType = 1.0;
	unsigned int numBatches=computeNumBatches(errorMode, returnPgram, *sizeData, numUniqueObjects, numFreqs, dataType);

	//compute longest time span of an object for paper
	// double maxspan=0;
	// unsigned int objectIdxmaxspan=0;
	// for (int i=0; i<numUniqueObjects; i++)
	// {
	// 	double timespan=timeX[objectLookup[i].idxMax]-timeX[objectLookup[i].idxMin]; 
	// 	if (timespan>maxspan)
	// 	{
	// 		maxspan=timespan;
	// 		objectIdxmaxspan=i;
	// 	}
	// }

	// printf("ObjId: %u, Max span: %f: ", objectLookup[objectIdxmaxspan].objId, maxspan);


	//copy pgram back to the host if enabled
	if(returnPgram == 1){
		*pgram=(DTYPE *)malloc(sizeof(DTYPE)*(uint64_t)numFreqs*(uint64_t)numUniqueObjects);
	}

	//allocate memory for the found periods
	*foundPeriod=(DTYPE *)malloc(sizeof(DTYPE)*numUniqueObjects);
	//allocate memory for the power corresponding to the period
	*foundPower=(DTYPE *)malloc(sizeof(DTYPE)*numUniqueObjects);

	//if there's only one batch then we execute the function as normal and do not partition the data
	if (numBatches==1){
		batchGPULS(objectId, timeX, magY, magDY, sizeData, minFreq, maxFreq, numFreqs, sumPeriods, *pgram, *foundPeriod, *foundPower);
	}else{ //if there is more than 1 batch


	//partition the total dataset
	//use cumulative sum of the length of the time series to perform partitioning
	unsigned int sumBatch=0;
	unsigned int totalSum=0; //sanity check
	unsigned int batchSize=*sizeData/numBatches;

	//batch indices into the main arrays
	unsigned int * dataIdxMin=(unsigned int *) malloc(sizeof(unsigned int)*numBatches);
	unsigned int * dataIdxMax=(unsigned int *) malloc(sizeof(unsigned int)*numBatches);

	//the number of data elems in each batch
	unsigned int * dataSizeBatches=(unsigned int *) malloc(sizeof(unsigned int)*numBatches);

	//the number of objects in the batch
	unsigned int * numObjectsInEachBatch=(unsigned int *) malloc(sizeof(unsigned int)*numBatches);

	//pgram write offset
	uint64_t * pgramWriteOffset=(uint64_t *) malloc(sizeof(uint64_t)*numBatches);

	//period write offset
	uint64_t * periodWriteOffset=(uint64_t *) malloc(sizeof(uint64_t)*numBatches);
	
	
	int batchCnt=0;
	unsigned int numObjectBatchCnt=0;
	
	for (unsigned int i=0; i<numUniqueObjects; i++)
	{
		totalSum+=objectLookup[i].idxMax-objectLookup[i].idxMin+1;
		sumBatch+=objectLookup[i].idxMax-objectLookup[i].idxMin+1;
		numObjectBatchCnt++;

		//if we reach the end of the batch, then we need to store the minimum and maximum ids in the array
		if (sumBatch>=batchSize)
		{
			numObjectsInEachBatch[batchCnt]=numObjectBatchCnt;
			dataIdxMax[batchCnt]=totalSum-1;
			sumBatch=0;	
			if (batchCnt>0)
			{
			dataIdxMin[batchCnt]=dataIdxMax[batchCnt-1]+1; 
			}
			batchCnt++;
			
			numObjectBatchCnt=0;
		}
	}
	//the first value is simply the first element
	dataIdxMin[0]=0;

	//the last data ranges will not get set in the loop
	dataIdxMin[numBatches-1]=dataIdxMax[numBatches-2]+1;
	dataIdxMax[numBatches-1]=objectLookup[numUniqueObjects-1].idxMax;

	//number of objects in each batch for allocating pgram memory
	numObjectsInEachBatch[numBatches-1]=numObjectBatchCnt;
	
	
	for (int i=0; i<numBatches; i++)
	{
		dataSizeBatches[i]=dataIdxMax[i]-dataIdxMin[i]+1;
		// printf("\nBatch %d, datasize: %u, number of objects: %u, dataIdxMin/Max: %u,%u", i, dataSizeBatches[i], numObjectsInEachBatch[i], dataIdxMin[i], dataIdxMax[i]);	
	}

	//cumulative sum for pgram write offset
	pgramWriteOffset[0]=0;
	periodWriteOffset[0]=0;
	uint64_t cumulativeObjects=0;
	for (int i=1; i<numBatches; i++)
	{
		cumulativeObjects+=numObjectsInEachBatch[i-1];
		pgramWriteOffset[i]=(uint64_t)cumulativeObjects*(uint64_t)numFreqs;
		periodWriteOffset[i]=cumulativeObjects;
		// printf("\nCumulative objects (batch: %d): %u",i,cumulativeObjects);
	}




	
	//parallelize using the number of GPU's threads (e.g., 2 GPUs use 2 threads)
	#pragma omp parallel for num_threads(numGPU) schedule(dynamic)
	for (int i=0; i<numBatches; i++)
	{

		int tid=omp_get_thread_num();
		cudaSetDevice(tid);
		unsigned int idxMin=dataIdxMin[i];

		DTYPE sumPeriodsBatch=0;
		// unsigned int idxMax=dataIdxMax[i];
		// printf("\nPeriod write offset, batch %d: %u",i,periodWriteOffset[i]);
		batchGPULS(&objectId[idxMin], &timeX[idxMin], &magY[idxMin], &magDY[idxMin], &dataSizeBatches[i], minFreq, maxFreq, numFreqs, &sumPeriodsBatch, *pgram+pgramWriteOffset[i], *foundPeriod+(periodWriteOffset[i]), *foundPower+(periodWriteOffset[i]));

		// printf("\nSum periods batch: %0.9f", sumPeriodsBatch);
		#pragma omp atomic
		*sumPeriods+=sumPeriodsBatch;
	}

	free(dataIdxMin);
	free(dataIdxMax);
	free(dataSizeBatches);
	free(numObjectsInEachBatch);
	free(pgramWriteOffset);

	}//end of else statement for numBatches>1



  	///////////////////////
  	//Output

	if(printPeriods == 1){
		//print found periods to stdout
		outputPeriodsToStdout(objectLookup, numUniqueObjects, *foundPeriod, *foundPower);
	}else if(printPeriods == 2){
		//print found periods to file
		outputPeriodsToFile(objectLookup, numUniqueObjects, *foundPeriod, *foundPower);
	}

  	//Output pgram to file
	if(printPgram == 1){
		outputPgramToFile(objectLookup, numUniqueObjects, numFreqs, pgram);  	
	}
  	
  	//End output
  	///////////////////////


	return;

}


template <typename DTYPE>
void outputPgramToFile(struct lookupObj * objectLookup, unsigned int numUniqueObjects, unsigned int numFreqs, DTYPE ** pgram)
{
	
  	printf("\nPrinting the pgram to file: %s", pgramOutFileName);
	ofstream pgramoutput;
	pgramoutput.open(pgramOutFileName,ios::out);	
  	pgramoutput.precision(4);
  	for (unsigned int i=0; i<numUniqueObjects; i++)
	{
		pgramoutput<<objectLookup[i].objId<<", ";
		for (int j=0; j<numFreqs; j++)
		{
		pgramoutput<<(*pgram)[(i*numFreqs)+j]<<", ";
		}
		pgramoutput<<endl;
	}
  	pgramoutput.close();
}



template <typename DTYPE>
void outputPeriodsToFile(struct lookupObj * objectLookup, unsigned int numUniqueObjects, DTYPE * foundPeriod, DTYPE * foundPower)
{
	// char fnamebestperiods[]= periodOutFileName;
  	printf("\nPrinting the best periods/found power to file: %s", periodOutFileName);
	ofstream bestperiodsoutput;
	bestperiodsoutput.open(periodOutFileName,ios::out);	
  	bestperiodsoutput.precision(6);
  	for (unsigned int i=0; i<numUniqueObjects; i++)
	{
		bestperiodsoutput<<objectLookup[i].objId<<", "<<foundPeriod[i]<<", "<<foundPower[i]<<endl;
	}
  	bestperiodsoutput.close();
}

template <typename DTYPE>
void outputPeriodsToFileTopThree(struct lookupObj * objectLookup, unsigned int numUniqueObjects, DTYPE * foundPeriod, DTYPE * foundPower)
{
	char fnamebestperiods[]="bestperiods_top_three.txt";
  	printf("\nPrinting the top three best periods/found powers to file (object id, period #1, period #2, period #3, power #1, power #2, power #3: %s", fnamebestperiods);
	ofstream bestperiodsoutput;
	bestperiodsoutput.open(fnamebestperiods,ios::out);	
  	bestperiodsoutput.precision(6);
  	for (uint64_t i=0; i<numUniqueObjects; i++)
	{
		uint64_t offset=i*(uint64_t)3;	
		bestperiodsoutput<<objectLookup[i].objId<<", "<<foundPeriod[offset]<<", "<<foundPeriod[offset+(uint64_t)1]<<", "<<foundPeriod[offset+(uint64_t)2]<<", "<<foundPower[offset]<<", "<<foundPower[offset+(uint64_t)1]<<", "<<foundPower[offset+(uint64_t)2]<<endl;
	}
  	bestperiodsoutput.close();
}

template <typename DTYPE>
void outputPeriodsToStdout(struct lookupObj * objectLookup, unsigned int numUniqueObjects, DTYPE * foundPeriod, DTYPE * foundPower)
{
	for (int i=0; i<numUniqueObjects; i++)
  	{
	  	printf("\nObject: %d Period: %f, Power: %f ",objectLookup[i].objId,foundPeriod[i], foundPower[i]);
  	}
}

//Send the minimum and maximum frequency and number of frequencies to test to the GPU (not a list of frequencies)
template <typename DTYPE>
void batchGPULS(unsigned int * objectId, DTYPE * timeX,  DTYPE * magY, DTYPE * magDY, unsigned int * sizeData, const DTYPE minFreq, const DTYPE maxFreq, const unsigned int numFreqs, DTYPE * sumPeriods, DTYPE * pgram, DTYPE * foundPeriod, DTYPE * foundPower)
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

    // foundPeriod=(DTYPE *)malloc(sizeof(DTYPE)*numUniqueObjects);



	//allocate memory on the GPU
	gpuErrchk(cudaMalloc((void**)&dev_timeX, sizeof(DTYPE)*(*sizeData)));
	gpuErrchk(cudaMalloc((void**)&dev_magY, sizeof(DTYPE)*(*sizeData)));

	//If astropy implementation with error
	DTYPE * dev_magDY;
	if(errorMode == 1){
		gpuErrchk(cudaMalloc((void**)&dev_magDY, sizeof(DTYPE)*(*sizeData)));

		//Need to incorporate error into magnitudes
		for (int i=0; i<numUniqueObjects; i++)
		{
			unsigned int idxMin=objectLookup[i].idxMin;
			unsigned int idxMax=objectLookup[i].idxMax;
			unsigned int sizeDataForObject=idxMax-idxMin+1;
			updateYerrorfactor(&magY[idxMin], &magDY[idxMin], sizeDataForObject);
		}	
	}

	DTYPE * pinned_buffer;
	unsigned int sizeBufferElems;
	if(returnPgram == 1){
		// Result periodogram must be number of unique objects * the size of the frequency array
		gpuErrchk(cudaMalloc((void**)&dev_pgram, sizeof(DTYPE)*numFreqs*numUniqueObjects));

		//Make a small pinned memory buffer for transferring the array back
		
		sizeBufferElems=(8*1024*1024)/(sizeof(DTYPE));
		gpuErrchk(cudaMallocHost((void**)&pinned_buffer, sizeof(DTYPE)*sizeBufferElems*nStreams));	
	} else {
		//If not returning the pgram then do not allocate memory
		dev_pgram=NULL;
	}

	//the maximum power in each periodogram. Use this when we don't want to return the periodogram
	gpuErrchk(cudaMalloc((void**)&dev_foundPeriod, sizeof(DTYPE)*numUniqueObjects));
	gpuErrchk(cudaMalloc((void**)&dev_sizeData, sizeof(unsigned int)));
	gpuErrchk(cudaMalloc((void**)&dev_objectLookup, sizeof(lookupObj)*(numUniqueObjects)));

	//copy data to the GPU
	gpuErrchk(cudaMemcpy( dev_timeX, timeX, sizeof(DTYPE)*(*sizeData), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy( dev_magY, magY, sizeof(DTYPE)*(*sizeData), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy( dev_sizeData, sizeData, sizeof(unsigned int), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy( dev_objectLookup, objectLookup, sizeof(lookupObj)*(numUniqueObjects), cudaMemcpyHostToDevice));

	if(errorMode == 1){
		gpuErrchk(cudaMemcpy( dev_magDY, magDY, sizeof(DTYPE)*(*sizeData), cudaMemcpyHostToDevice));
	}


	
	const int numBlocks=numUniqueObjects;
	double tstart=omp_get_wtime();
  	//Do lomb-scargle
	if(returnPgram == 1){
		if(errorMode == 0){
			lombscargleBatchReturnPgram<<< numBlocks, BLOCKSIZE>>>(dev_timeX, dev_magY, dev_objectLookup, dev_pgram, dev_foundPeriod, minFreq, maxFreq, numFreqs);

		}else{
			lombscargleBatchError<<< numBlocks, BLOCKSIZE>>>(dev_timeX, dev_magY, dev_magDY, dev_objectLookup, dev_pgram, dev_foundPeriod, minFreq, maxFreq, numFreqs);
		}
	} else{
		if(errorMode == 0){
			lombscargleBatch<<< numBlocks, BLOCKSIZE>>>(dev_timeX, dev_magY, dev_objectLookup, dev_pgram, dev_foundPeriod, minFreq, maxFreq, numFreqs);

		}else{
			lombscargleBatchError<<< numBlocks, BLOCKSIZE>>>(dev_timeX, dev_magY, dev_magDY, dev_objectLookup, dev_pgram, dev_foundPeriod, minFreq, maxFreq, numFreqs);
		}
	}



  	cudaDeviceSynchronize();

  	double tend=omp_get_wtime();
  	printf("\nTime to compute kernel: %f",tend-tstart);


  	

  	//copy pgram back to the host if enabled
	unsigned int sizeDevData;
	if(returnPgram == 1){
		sizeDevData=numUniqueObjects*numFreqs;
		pinnedMemoryCopyDtoH(pinned_buffer, sizeBufferElems, dev_pgram, pgram, sizeDevData);
	} else {
		//Return the maximum power for each object
		gpuErrchk(cudaMemcpy( foundPeriod, dev_foundPeriod, sizeof(DTYPE)*(numUniqueObjects), cudaMemcpyDeviceToHost));
	}


  	//For each object, find the maximum power in the pgram
	
	if(returnPgram == 1){
		//compute the maximum power using the returned pgram
		double tstartcpupgram=omp_get_wtime();
		printf("\nCompute period from pgram on CPU:");
  

		if(normalizePgram == 1 && errorMode ==0){
			normalizepgram(pgram, objectLookup, magY, numUniqueObjects, numFreqs);
		}
  
  
		double freqStep=(maxFreq-minFreq)/(numFreqs*1.0);	
		#pragma omp parallel for num_threads(nThreadsCpu)
		for (uint64_t i=0; i<numUniqueObjects; i++)
		{
			DTYPE maxPower=0;
			uint64_t maxPowerIdx=0;
			for (uint64_t j=0; j<(uint64_t)numFreqs; j++)
			{
				if (maxPower<pgram[i*(uint64_t)numFreqs+j])
				{
					maxPower=pgram[i*(uint64_t)numFreqs+j];
					maxPowerIdx=j;
				}
			}
  
			//Validation: total period values
		  foundPeriod[i]=(1.0/(minFreq+(maxPowerIdx*freqStep)))*2.0*M_PI;
		  foundPower[i]=maxPower;
			
				
			// if (i==0)
			// {
			// 	printf("\nmaxPowerIdx: %u", maxPowerIdx);
			// }
		}
  
		double tendcpupgram=omp_get_wtime();
		printf("\nTime to compute the periods on the CPU using the pgram: %f", tendcpupgram - tstartcpupgram);
	}else{
		printf("\nCompute period in the kernel directly bypassing the pgram array");

	}

  	
  	//for validation
	for (unsigned int i=0; i<numUniqueObjects; i++)
  	{
	  	(*sumPeriods)+=foundPeriod[i];
  	}


  	
  	

	

  	//free memory
  	// free(foundPeriod);
  	free(objectLookup);

  	//free memory-- CUDA
  	cudaFree(dev_timeX);
  	cudaFree(dev_magY);
	cudaFree(dev_sizeData);
	cudaFree(dev_pgram);
	cudaFree(dev_foundPeriod);
	cudaFree(dev_objectLookup);

	if(errorMode == 1){
		cudaFree(dev_magDY); 
	}


	if(returnPgram == 1){
		cudaFreeHost(pinned_buffer);
	}


	
	


  	


}

// double computeMeanDataSquared(DTYPE * magY, uint64_t sizeData)
// {
// 	//compute the mean of the data squared
// 	double sum=0;
// 	for (uint64_t k=0; k<sizeData; k++)
// 	{
// 		sum+=(magY[k]*magY[k]);
// 	}

// 	double meandatasquared=sum/(sizeData*1.0);
// 	return meandatasquared;
// }

template <typename DTYPE>
double computeStandardDevSquared(DTYPE * magY, uint64_t sizeData)
{

	//Step 1: compute the mean
	double sum=0;
	for (uint64_t k=0; k<sizeData; k++)
	{
		sum+=magY[k];
	}

	double mean=sum/(sizeData*1.0);

	//Step 2: compute the standard deviation
	double sum2=0;	
	for (uint64_t k=0; k<sizeData; k++)
	{
		sum2+=(magY[k]-mean)*(magY[k]-mean);
	}

	double sigma=sqrt(sum2/(sizeData*1.0));
	
	return sigma*sigma;
}

template <typename DTYPE>
void normalizepgram(DTYPE * pgram, struct lookupObj * objectLookup, DTYPE * magY, uint64_t numUniqueObjects, uint64_t numFreqs)
{

		
	#pragma omp parallel for num_threads(nThreadsCpu)
  	for (uint64_t i=0; i<numUniqueObjects; i++)
  	{
  		//get the data size for the object
		uint64_t idxMin=objectLookup[i].idxMin;
		uint64_t idxMax=objectLookup[i].idxMax;
		uint64_t sizeDataForObject=idxMax-idxMin+1;
		
		double stddevsquared=computeStandardDevSquared(magY+idxMin, sizeDataForObject);

	  	for (uint64_t j=0; j<numFreqs; j++)
	  	{
	  		pgram[i*numFreqs+j]*=2.0/(sizeDataForObject*stddevsquared);	
	  	}
  	}

}

template <typename DTYPE>
void normalizepgramsingleobject(DTYPE * pgram, DTYPE * magY, uint64_t sizeDataForObject, uint64_t numFreqs)
{
		
		//compute the mean of the data squared
		double stddevsquared=computeStandardDevSquared(magY, sizeDataForObject);

  		#pragma omp parallel for num_threads(nThreadsCpu)
	  	for (uint64_t j=0; j<numFreqs; j++)
	  	{
	  		pgram[j]*=2.0/(sizeDataForObject*stddevsquared);	
	  	}
}




//parallelize over the frequency if computing a single object
template <typename DTYPE>
void lombscargleCPUOneObject(DTYPE * timeX,  DTYPE * magY, unsigned int * sizeData, const DTYPE minFreq, const DTYPE maxFreq, const unsigned int numFreqs, DTYPE * foundPeriod, DTYPE * foundPower, DTYPE * pgram)
{
	pgram=(DTYPE *)malloc(sizeof(DTYPE)*(numFreqs));

	const DTYPE freqStep=(maxFreq-minFreq)/(numFreqs*1.0);	

	//1 refers to the mode of executing in parallel inside the LS algorithm
	lombscarglecpu(1, timeX, magY, *sizeData, numFreqs, minFreq, maxFreq, freqStep, pgram);	

	computePeriod(pgram, numFreqs, minFreq, freqStep, foundPeriod, foundPower);

	if(normalizePgram == 1 && errorMode == 0){
		normalizepgramsingleobject(pgram, magY, *sizeData, numFreqs);
	}

	if(printPeriods == 1){
		printf("\nPeriod: %f", *foundPeriod);
	}

}

//uses error propogation
template <typename DTYPE>
void lombscargleCPUOneObjectError(DTYPE * timeX,  DTYPE * magY, DTYPE * magDY, unsigned int * sizeData, const DTYPE minFreq, const DTYPE maxFreq, const unsigned int numFreqs, DTYPE * foundPeriod, DTYPE * foundPower, DTYPE * pgram)
{
	pgram=(DTYPE *)malloc(sizeof(DTYPE)*(numFreqs));

	const DTYPE freqStep=(maxFreq-minFreq)/(numFreqs*1.0);	

	//1 refers to the mode of executing in parallel inside the LS algorithm
	lombscarglecpuError(1, timeX, magY, magDY, *sizeData, numFreqs, minFreq, maxFreq, freqStep, pgram);	

	computePeriod(pgram, numFreqs, minFreq, freqStep, foundPeriod, foundPower);
	
	if(normalizePgram == 1 && errorMode == 0){
		normalizepgramsingleobject(pgram, magY, *sizeData, numFreqs);
	}

	if(printPeriods == 1){
		printf("\nPeriod: %f", *foundPeriod);
	}

}

template <typename DTYPE>
void lombscargleCPUBatchError(unsigned int * objectId, DTYPE * timeX,  DTYPE * magY, DTYPE * magDY, unsigned int * sizeData, const DTYPE minFreq, const DTYPE maxFreq, const unsigned int numFreqs, DTYPE * sumPeriods, DTYPE * pgram, DTYPE * foundPeriod, DTYPE * foundPower)
{


	//compute the object ranges in the arrays and store in struct
	//This is given by the objectId
	struct lookupObj * objectLookup=NULL;
	unsigned int numUniqueObjects;
	computeObjectRanges(objectId, sizeData, &objectLookup, &numUniqueObjects);
	pgram=(DTYPE *)malloc(sizeof(DTYPE)*(numFreqs)*numUniqueObjects);
	foundPeriod=(DTYPE *)malloc(sizeof(DTYPE)*numUniqueObjects);
	foundPower=(DTYPE *)malloc(sizeof(DTYPE)*numUniqueObjects);

	const DTYPE freqStep=(maxFreq-minFreq)/(numFreqs*1.0);	

	//for each object, call the sequential cpu algorithm
	#pragma omp parallel for num_threads(nThreadsCpu) schedule(dynamic)
	for (int i=0; i<numUniqueObjects; i++)
	{
		unsigned int idxMin=objectLookup[i].idxMin;
		unsigned int idxMax=objectLookup[i].idxMax;
		unsigned int sizeDataForObject=idxMax-idxMin+1;
		uint64_t pgramWriteOffset=(uint64_t)i*(uint64_t)numFreqs;
		//0 refers to the mode of executing sequentially inside the LS algorithm
		lombscarglecpuError(0, &timeX[idxMin], &magY[idxMin], &magDY[idxMin], sizeDataForObject, numFreqs, minFreq, maxFreq, freqStep, pgram+pgramWriteOffset);	

		if(normalizePgram == 1 && errorMode == 0){
			normalizepgramsingleobject(pgram+pgramWriteOffset, &magY[idxMin], sizeDataForObject, numFreqs);
		}

		computePeriod(pgram+pgramWriteOffset, numFreqs, minFreq, freqStep, &foundPeriod[i], &foundPower[i]);
	}

	

	///////////////////////
  	//Output

	if(printPeriods == 1){
		//print found periods to stdout
		outputPeriodsToStdout(objectLookup, numUniqueObjects, foundPeriod, foundPower);
	}else if(printPeriods == 2){
		//print found periods to file
		outputPeriodsToFile(objectLookup, numUniqueObjects, foundPeriod, foundPower);
	}

  	
  	//Output pgram to file
	if(printPgram ==1){
		outputPgramToFile(objectLookup, numUniqueObjects, numFreqs, &pgram);  	
	}
  	
  	//End output
  	///////////////////////

	//Validation
 	for (int i=0; i<numUniqueObjects; i++)
  	{
	  	(*sumPeriods)+=foundPeriod[i];
  	}

	

}

template <typename DTYPE>
void lombscargleCPUBatch(unsigned int * objectId, DTYPE * timeX,  DTYPE * magY, unsigned int * sizeData, const DTYPE minFreq, const DTYPE maxFreq, const unsigned int numFreqs, DTYPE * sumPeriods, DTYPE * pgram, DTYPE * foundPeriod, DTYPE * foundPower)
{


	//compute the object ranges in the arrays and store in struct
	//This is given by the objectId
	struct lookupObj * objectLookup=NULL;
	unsigned int numUniqueObjects;
	computeObjectRanges(objectId, sizeData, &objectLookup, &numUniqueObjects);
	pgram=(DTYPE *)malloc(sizeof(DTYPE)*(numFreqs)*numUniqueObjects);
	foundPeriod=(DTYPE *)malloc(sizeof(DTYPE)*numUniqueObjects);
	foundPower=(DTYPE *)malloc(sizeof(DTYPE)*numUniqueObjects);

	const DTYPE freqStep=(maxFreq-minFreq)/(numFreqs*1.0);	

	//for each object, call the sequential cpu algorithm
	#pragma omp parallel for num_threads(nThreadsCpu) schedule(dynamic)
	for (int i=0; i<numUniqueObjects; i++)
	{
		unsigned int idxMin=objectLookup[i].idxMin;
		unsigned int idxMax=objectLookup[i].idxMax;
		unsigned int sizeDataForObject=idxMax-idxMin+1;
		uint64_t pgramWriteOffset=(uint64_t)i*(uint64_t)numFreqs;
		//0 refers to the mode of executing sequentially inside the LS algorithm
		lombscarglecpu(0, &timeX[idxMin], &magY[idxMin], sizeDataForObject, numFreqs, minFreq, maxFreq, freqStep, pgram+pgramWriteOffset);	

		if(normalizePgram ==1 && errorMode == 0){
			normalizepgramsingleobject(pgram+pgramWriteOffset, &magY[idxMin], sizeDataForObject, numFreqs);
		}
		
		computePeriod(pgram+pgramWriteOffset, numFreqs, minFreq, freqStep, &foundPeriod[i], &foundPower[i]);
	}


	///////////////////////
  	//Output
  	
	if(printPeriods == 1){
		//print found periods to stdout
		outputPeriodsToStdout(objectLookup, numUniqueObjects, foundPeriod, foundPower);
	}else if(printPeriods == 2){
		//print found periods to file
		outputPeriodsToFile(objectLookup, numUniqueObjects, foundPeriod, foundPower);
	}

  	
  	//Output pgram to file
	if(printPgram ==1){
		outputPgramToFile(objectLookup, numUniqueObjects, numFreqs, &pgram);  	
	}

  	//End output
  	///////////////////////



	//Validation
 	for (int i=0; i<numUniqueObjects; i++)
  	{
	  	(*sumPeriods)+=foundPeriod[i];
  	}

	

}

template <typename DTYPE>
void computePeriod(DTYPE * pgram, const unsigned int numFreqs, const DTYPE minFreq, const DTYPE freqStep, DTYPE * foundPeriod, DTYPE * foundPower)
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
		*foundPower=maxPower;	  	
}

template <typename DTYPE>
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
template <typename DTYPE>
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
			#pragma omp parallel for num_threads(nThreadsCpu) schedule(static)
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
template <typename DTYPE>
void updateYerrorfactor(DTYPE * y, DTYPE *dy, const unsigned int sizeData)
{

		//Pre-center the data with error:
		//w = dy ** -2
		//sum w
		DTYPE * w =(DTYPE *)malloc(sizeof(DTYPE)*sizeData);
		DTYPE sumw=0;
		#pragma omp parallel for num_threads(nThreadsCpu) reduction(+:sumw)
		for (int i=0; i<sizeData; i++)
		{
			w[i]=1.0/sqrt(dy[i]);
			sumw+=w[i];
		}
		//compute dot product w,y
		DTYPE dotwy=0;
		#pragma omp parallel for num_threads(nThreadsCpu) reduction(+:dotwy)
		for (int i=0; i<sizeData; i++)
		{
			dotwy+=w[i]*y[i];
		}

		//update y to account for dot product and sum w
		//y = y - dot(w, y) / np.sum(w)	
		#pragma omp parallel for num_threads(nThreadsCpu)
		for (int i=0; i<sizeData; i++)
		{
			y[i]=y[i]-dotwy/sumw;
		}

		free(w);
}

//lombsscarge on the CPU for AstroPy with error
//Mode==0 means run sequentially (batch mode)
//Mode==1 means parallelize over the frequency loop (multiobject)
template <typename DTYPE>
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
			
			#pragma omp parallel for num_threads(nThreadsCpu) schedule(static)
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
template <typename DTYPE>
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


template <typename DTYPE>
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


	if(errorMode == 0){
		*sizeData=(unsigned int)tmpAllData.size()/3;
	}else{
		*sizeData=(unsigned int)tmpAllData.size()/4;
	}
  	printf("\nData import: Total rows: %u",*sizeData);
  	
  	*objectId=(unsigned int *)malloc(sizeof(DTYPE)*(*sizeData));
  	*timeX=   (DTYPE *)malloc(sizeof(DTYPE)*(*sizeData));
  	*magY=    (DTYPE *)malloc(sizeof(DTYPE)*(*sizeData));


	if(errorMode == 0){
		for (int i=0; i<*sizeData; i++){
			(*objectId)[i]=tmpAllData[(i*3)+0];
			(*timeX)[i]   =tmpAllData[(i*3)+1];
			(*magY)[i]    =tmpAllData[(i*3)+2];
		}
	}else{
		*magDY=   (DTYPE *)malloc(sizeof(DTYPE)*(*sizeData));
		for (int i=0; i<*sizeData; i++){
			(*objectId)[i]=tmpAllData[(i*4)+0];
			(*timeX)[i]   =tmpAllData[(i*4)+1];
			(*magY)[i]    =tmpAllData[(i*4)+2];
			(*magDY)[i]    =tmpAllData[(i*4)+3];
	  	}
	}

}




void warmUpGPU(){
printf("\nLoad CUDA runtime (initialization overhead)\n");

for (int i=0; i<numGPU; i++)
{
cudaSetDevice(i); 	
cudaDeviceSynchronize();
}

}


double getGPUCapacity()
{
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  
  //Read the global memory capacity from the device.
  unsigned long int globalmembytes=0;
  gpuErrchk(cudaMemGetInfo(NULL,&globalmembytes));
  double totalcapacityGiB=globalmembytes*1.0/(1024*1024*1024.0);

  printf("\n[Device name: %s, Detecting GPU Global Memory Capacity] Size in GiB: %f", prop.name, totalcapacityGiB);
  double underestcapacityGiB=totalcapacityGiB*0.50; //alpha of 0.50
  printf("\n[Underestimating GPU Global Memory Capacity] Size in GiB: %f", underestcapacityGiB);
  return underestcapacityGiB;
}