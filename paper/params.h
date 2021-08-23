#define NTHREADSCPU 16 //used for parallelizing GPU tasks and the number of threads used in the CPU implementations
#define DTYPE double //float or double
#define BLOCKSIZE 512 //CUDA block size

#define NUMGPU 1 //the number of GPUs
				//multiple GPUs for batch mode only
					

#define RETURNPGRAM 1 //0-only computes the power on the GPU and doesn't return the pgram for each object
					  //1- returns the pgram for each object

#define ERROR 1 //0- standard L-S
				//1- Based on AstroPy with error propogation and fitted mean 
				//The astropy port uses the default configuration (only global memory, returns the pgram)
				//And it normalizes the pgram using the standard normalization
				//as documented here: https://docs.astropy.org/en/stable/timeseries/lombscargle.html#periodogram-algorithms
#define PINNED 1 //0- when returning the pgram (RETURNPGRAM==1) copy using standard pageable memory
				 //1- when returning the pgram (RETURNPGRAM==1) copy using pinned memory with SIZEPINNEDBUFFERMIB and NSTREAMS

#define SIZEPINNEDBUFFERMIB 8 //[WHEN PINNED==1] When returning the pgram, need to transfer using pinned memory. Use this many MiB per buffer
#define NSTREAMS 3 //[WHEN PINNED==1] When returning the pgram use this many streams each of size SIZEPINNEDBUFFERMIB

#define SHMEM 0 //Use shared memory for manually paging the input time and mag (for batch or single object)
			//0- use global memory for this
			//1- use shared memory for this

#define PRINTPERIODS 2  //0- do not print periods to stdout
						//1- print found periods to stdout 
						//2- print found periods to file (bestperiods.txt) [batch modes only]
						//3- print top 3 period solutions and their powers to file (bestperiods_top3.txt) [batch modes only] 
						
#define PRINTPGRAM 0 //0- do not print pgram
					 //1- print pgram to file (pgram.txt)

#define ALPHA 0.35 //used to underestimate the total global memory on the GPU
				  //batching scheme assumes all objects have the same number of observations
				  //for load balancing
				  //but, this assumption can cause the GPU to run out of global memory
				  
#define NORMALIZEPGRAM 1 //0-unnormalized pgram when using standard L-S (without error)
						//1- normalize the pgram when using standard L-S (without error) by power *= 2 / (N * mag.std() ** 2)
						
						//see: https://jakevdp.github.io/blog/2015/06/13/lomb-scargle-in-python/
						//When using the GPU, only works when returning the pgram to the host 

//Implementation with ERROR=0 (standard L-S)
//Can use global or shared memory, option to return the pgram

//Implementation with ERROR=1 (AstroPy)
//Has floating mean and error propogation
//Uses global memory kernel and returns the pgram

