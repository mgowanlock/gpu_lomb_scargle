#define NTHREADSCPU 16 //used for parallelizing GPU tasks and the number of threads used in the CPU implementations
#define DTYPE double //float or double
#define BLOCKSIZE 512 //CUDA block size

#define RETURNPGRAM 1 //0-only computes the power on the GPU and doesn't return the pgram for each object
					  //1- returns the pgram for each object

#define PINNED 1 //0- when returning the pgram (RETURNPGRAM==1) copy using standard pageable memory
				 //1- when returning the pgram (RETURNPGRAM==1) copy using pinned memory with SIZEPINNEDBUFFERMIB and NSTREAMS

#define SIZEPINNEDBUFFERMIB 8 //[WHEN PINNED==1] When returning the pgram, need to transfer using pinned memory. Use this many MiB per buffer
#define NSTREAMS 3 //[WHEN PINNED==1] When returning the pgram use this many streams each of size SIZEPINNEDBUFFERMIB

#define SHMEM 0 //Use shared memory for manually paging the input time and mag (for batch or single object)
			//0- use global memory for this
			//1- use shared memory for this

#define PRINTPERIODS 0 //print found periods to stdout							