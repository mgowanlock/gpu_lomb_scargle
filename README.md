# Lomb-Scargle Algorithm on the GPU (accompanying paper under review).

Code authors: Mike Gowanlock and Brian Donnelly

## There are three directories:
* data
* paper
* release

The data directory includes test data (all datasets used in the paper). The paper directory contains the source code used for the experimental evaluation in the paper. And the release code includes both CUDA and OpenACC functionality. The difference between the paper and release code is that many of the GPU performance parameters have been selected for the user so that a reasonable default configuration can be used without extensive knowledge of the details in the paper. However, if the user is interested in all of the bells and whistles included in the paper, then they should use the paper implementation.

## OpenACC
OpenACC allows the code to be executed on CPUs and GPUs, including offloading to both Nvidia and AMD GPUs. This implementation was included so that the algorithm can be used across different platforms; however, we do not claim that it is as efficient as the CUDA implementation.

## Data:
  * Ida 243 in the paper from the ZTF public survey (243_normalized_ztf_filter2.txt)
  * A batch of synthetic objects using a log-normal distribution from the paper (normalized_alltargs.200724_1_log_normal_obs.dat)
  * A single synthetic object with 3,555 measurements from the paper (8205_normalized.txt)

## Modes:
As described in the paper, the GPU algorithm allows for both a single object to be processed (e.g., a user wants to process a large time series or a large number of frequencies need to be searched). And it also allows for a batch of objects to be processed (e.g., deriving periods for an entire astronomical catalog, or near real-time period finding for ZTF or LSST during nighttime observing). 

## Makefile
A makefile has been included for each implementation. Make sure to update the compute capability flag to ensure you compile for the correct architecture. To find out which compute capability your GPU has, please refer to the compute capability table on Wikipedia: https://en.wikipedia.org/wiki/CUDA.

## Running the program using the paper implementation:
After compiling the computer program, you must enter the following command line arguments:
\<dataset file name\> \<minimum frequency\> \<maximum frequency\> \<number of frequencies to search\> \<mode\>
  
The paper implementation has the following modes:
* 1- GPU to process a batch of objects
* 2- GPU to process a single object
* 4- CPU to process a batch of objects
* 5- CPU to process a single object

Example for the batch mode:
$ ./main ../data/normalized_alltargs.200724_1_log_normal_obs.dat 1.005 150.796 1000000 1

Load CUDA runtime (initialization overhead)

Dataset file: ../data/normalized_alltargs.200724_1_log_normal_obs.dat
Minimum Frequency: 1.005000
Maximum Frequency: 150.796000
Number of frequencies to test: 1000000
Mode: 1
Data import: Total rows: 165221
Unique objects in file: 999
Time to compute kernel: 24.830843
Compute period from pgram on CPU:
Time to compute the periods on the CPU using the pgram: 0.166939
Total time to compute batch: 25.908213
[Validation] Sum of all periods: 716.437798

Example for the single object mode:
$ ./main ../data/243_normalized_ztf_filter2.txt 3.142 150.796 1000000 2

Load CUDA runtime (initialization overhead)

Dataset file: ../data/243_normalized_ztf_filter2.txt
Minimum Frequency: 3.142000
Maximum Frequency: 150.796000
Number of frequencies to test: 1000000
Mode: 2
Data import: Total rows: 29
Period: 0.096536
Time to compute kernel: 0.005606
Maximum power at found period: 1.080289
Total time to compute batch: 0.028207
[Validation] Period: 0.096536

## Running the program using the release implementation (CUDA GPU):
XXX Brian

## Running the program using the release implementation (OpenACC):
XXX Brian
