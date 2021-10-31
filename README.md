# Lomb-Scargle Algorithm on the GPU 

Code authors: Mike Gowanlock and Brian Donnelly

# Paper

If you use this software, please cite our paper below.

M. Gowanlock, D. Kramer, D.E. Trilling, N.R. Butler, B. Donnelly (2021)\
*Fast period searches using the Lomb–Scargle algorithm on Graphics Processing Units for large datasets and real-time applications*\
Astronomy and Computing, Elsevier\
https://doi.org/10.1016/j.ascom.2021.100472

## Python Interface
If you would like to skip all of the details regarding the C/CUDA implementation and configuration options and go directly to using the Python interface, please scroll down to the Python interface section below. 

## There are three directories:
* data
* source
* release

The data directory includes test data (all datasets used in the paper, in addition to other datasets for testing). The source directory contains the source code used for the experimental evaluation in the paper. And the release code includes both CUDA and OpenACC functionality. The difference between the paper and release code is that many of the GPU performance parameters have been selected for the user so that a reasonable default configuration can be used without extensive knowledge of the details in the paper. However, if the user is interested in all of the bells and whistles included in the paper, then they should use the paper implementation.

Note: We have updated the paper verson of the code base to include multi-GPU functionality. This functionality was not described in the paper. Additionally, information on the new Python interface is below



## Data:
The following datasets were used in the paper:
  * Ida 243 in the paper from the ZTF public survey (243_normalized_ztf_filter2.txt)
  * A single synthetic object with 3,555 measurements from the paper (8205_normalized.txt, 8205_normalized_with_error.txt)
  * A batch of synthetic objects using a log-normal distribution from the paper (normalized_alltargs.200724_1_log_normal_obs.dat)
  * Period solutions for the synthetic objects above, given in the column "lcper" (simobj.200724_1.dat)

Additional datasets have been used in the Python test program:
 * SDSS_stripe82_band_z.txt

## Modes: Single Object and Batched 
As described in the paper, the GPU algorithm allows for both a single object to be processed (e.g., a user wants to process a large time series or a large number of frequencies need to be searched). And it also allows for a batch of objects to be processed (e.g., deriving periods for an entire astronomical catalog, or near real-time period finding for ZTF or LSST during nighttime observing). 

## Modes: Standard L-S and Generalized Periodogram (Photometric Error/Floating Mean)
We include two versions. The Standard L-S algorithm that is found in SciPy which does not include photometric errors on the magnitudes is described in great detail throughout the paper. We also include the standard configuration from AstroPy which floats the mean and includes errors photometric errors. 

The dataset file for the standard L-S algorithm should be in the format: object id, time, mag. 

The dataset file for the generalized algorithm should be in the format: object id, time, mag, dmag. 

## Makefile
A makefile has been included for each implementation. Make sure to update the compute capability flag to ensure you compile for the correct architecture. To find out which compute capability your GPU has, please refer to the compute capability table on Wikipedia: https://en.wikipedia.org/wiki/CUDA.

## Running the program using the paper implementation (Standard L-S):
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

Dataset file: ../data/normalized_alltargs.200724_1_log_normal_obs.dat\
Minimum Frequency: 1.005000\
Maximum Frequency: 150.796000\
Number of frequencies to test: 1000000\
Mode: 1\
Data import: Total rows: 165221\
Unique objects in file: 999\
Time to compute kernel: 24.830843\
Compute period from pgram on CPU:\
Time to compute the periods on the CPU using the pgram: 0.166939\
Total time to compute batch: 25.908213\
[Validation] Sum of all periods: 716.437798


Example for the single object mode:
$ ./main ../data/243_normalized_ztf_filter2.txt 3.142 150.796 1000000 2

Load CUDA runtime (initialization overhead)

Dataset file: ../data/243_normalized_ztf_filter2.txt\
Minimum Frequency: 3.142000\
Maximum Frequency: 150.796000\
Number of frequencies to test: 1000000\
Mode: 2\
Data import: Total rows: 29\
Period: 0.096536\
Time to compute kernel: 0.005606\
Maximum power at found period: 1.080289\
Total time to compute batch: 0.028207\
[Validation] Period: 0.096536

## Running the program using the paper implementation (AstroPy Default with Error and Floating Mean):
The default implementation in AstroPy uses the generalized periodogram which floats the mean and includes photmetric errors. To run this version of the code use the same command line input parameters but use a dataset file that includes photometric errors. Make sure to compile using the "ERROR=1" flag.

Example for the batch mode:
$ ./main ../data/normalized_alltargs.200724_1_log_normal_obs_with_error.dat 1.00531 150.796 1000000 1

Load CUDA runtime (initialization overhead)

Dataset file: ../data/normalized_alltargs.200724_1_log_normal_obs_with_error.dat\
Minimum Frequency: 1.005310\
Maximum Frequency: 150.796000\
Number of frequencies to test: 1000000\
Mode: 1\
Executing L-S variant from AstroPy that propogates error and floats the mean\
Data import: Total rows: 165221\
Unique objects in file: 999\
Time to compute kernel: 8.583307\
Compute period from pgram on CPU:\
Time to compute the periods on the CPU using the pgram: 0.195461\
Total time to compute batch: 9.872271\
[Validation] Sum of all periods: 718.249253

Example for the single object mode:
$ ./main ../data/8205_normalized_with_error.txt 3.14159 150.79645 1000000 2

Load CUDA runtime (initialization overhead)

Dataset file: ../data/8205_normalized_with_error.txt\
Minimum Frequency: 3.141590\
Maximum Frequency: 150.796450\
Number of frequencies to test: 1000000\
Mode: 2\
Executing L-S variant from AstroPy that propogates error and floats the mean\
Data import: Total rows: 3555\
Period: 0.564736\
Time to compute kernel: 0.152766\
Maximum power at found period: 0.712905\
Total time to compute batch: 0.213529\
[Validation] Period: 0.564736


## Running the program using the release implementation (CUDA GPU):

After compiling the computer program, you must use the flags:
-f < path to data> -min < min frequency> -max < max frequency> -fq < Num frequencies to test> -m < Compute Mode (1-4)>

Note: For a full list of flags use ./main --help

Example using floats and errors on batch of objects:

 ./main -f ../data/normalized_alltargs.200724_1_log_normal_obs_with_error.dat -min 1.005 -max 150.796 -fq 10000 -m 1

Load CUDA runtime (initialization overhead)

Dataset file: ../data/normalized_alltargs.200724_1_log_normal_obs_with_error.dat
Minimum Frequency: 1.005000
Maximum Frequency: 150.796000
Number of frequencies to test: 10000

Running with data type of float for better performance at cost of accuracy.
Set argument 5 to 1 to increase accuracy at cost of performance

Printing is turned off. To print the period of every object to screen, set argument 6 to 1.

Executing L-S variant from AstroPy that propogates error and floats the mean.
This can be disabled by setting argument 7 to 0.

Data import: Total rows: 165221
Mode: 1 Detected [Processing a batch of objects]
Unique objects in file: 999
Time to compute kernel: 0.175216
Compute period from pgram on CPU:
Time to compute the periods on the CPU using the pgram: 0.008500
Total time to compute batch: 0.469490
[Validation] Sum of all periods: 722.733643


Example using doubles with no error on a single object: 

./main -f ../data/8205_normalized.txt -min 3.14159 -max 150.79645 -fq 10000 -m 4

Load CUDA runtime (initialization overhead)

Dataset file: ../data/8205_normalized.txt
Minimum Frequency: 3.141590
Maximum Frequency: 150.796450
Number of frequencies to test: 10000

Running with data type of double for better precision at cost of performance.
Set argument 5 to 0 to increase performance at cost of accuracy

Printing is turned off. To print the period of every object to screen, set argument 6 to 1.

Executing L-S without errors, to turn on; set argument 7 to 1

Data import: Total rows: 3555
Period: 0.564541
Time to compute kernel: 0.018303
Total time to compute period: 0.030021
[Validation] Period: 0.564541

## Python Interface
The Python interface calls a C shared library of the compiled sources. You can either build the shared libraries or use the precompiled shared libraries. We recommend building them from source, because there are a couple of machine specific parameters that should be set.

To build the shared libraries from source, open the Makefile in the source directory and modify the following variables (the number of GPUs and the number of physical cores on your machine). 
```
-DNUMGPU=1
-DNTHREADSCPU=16
```

Next, run the following command, which will build the shared libraries. Note that these binaries are large, as they are compiled to maximize GPU compatibility, and thus are compiled for the following compute capabilities: 6.0, 6.1, 7.0, 7.2, 7.5, 8.0 (these encompass Pascal, Volta, Turing, and Ampere architectures). We have tested that these binaries work on Pascal (6.1) and Turing (7.5) architectures.

```
$make make_python_shared_libs
```

This will generate four shared libraries: libpylserrordouble.so, libpylserrorfloat.so, libpylsnoerrordouble.so, and libpylsnoerrorfloat.so. If you do not want to compile these, copies have been provided in the repository.

Make sure to store your shared libraries in a location where they will be searched/accessible by Python. To eliminate issues with this, we have set up our Python test program to search its directory for the shared libraries. 

Now, to use the Python interface, there are two files. The main module is lombscarglegpu.py, which is imported by the test example program, lombscargle_test_example.py. 

The Python module contains a few additional options over the C interface. 
* It allows you to hide the output of the C shared libraries by setting the verbose mode flag to false.
* If you are unsure of the number of frequencies to use, it employs the method by Richards et al. (2011) for selecting the number of frequencies.
* By default, it will use 32 bit floating point precision (FP32) over FP64. This is because many consumer grade GPUs have limited hardware dedicated to FP64. However, the user may wish to change this to FP64.

For each object, the Python module will return the period at the highest peak in the periodogram, and the periodogram itself.

The output of lombscargle_test_example.py is stored in python_output_lomb_scargle_test_example.txt in the python_interface_output directory.




