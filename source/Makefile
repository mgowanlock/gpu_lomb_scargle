
#####################################
#Build binaries for the code in the paper
#see params.h for the parameters
SOURCES = main.cu 
CUDAOBJECTS = main.o kernel.o
CC = nvcc
EXECUTABLE = main

#select the compute capability of your GPU
COMPUTE_CAPABILITY = 75

FLAGS =  -std=c++14 -Xcompiler "-W" -O3 -Xcompiler -fopenmp -use_fast_math -lcuda -lineinfo 

#used for the C version of the code
COMPUTE_CAPABILITY_FLAGS = -arch=compute_$(COMPUTE_CAPABILITY) -code=sm_$(COMPUTE_CAPABILITY)
CFLAGS = -c 

#used for the Python wrapper to maximize GPU compatibility
CUDAARCHFLAGS = -arch=compute_$(COMPUTE_CAPABILITY) \
-gencode=arch=compute_60,code=sm_60 \
-gencode=arch=compute_61,code=sm_61 \
-gencode=arch=compute_70,code=sm_70 \
-gencode=arch=compute_72,code=sm_72 \
-gencode=arch=compute_75,code=sm_75 \
-gencode=arch=compute_80,code=sm_80 \
#put future generations here.

all: $(EXECUTABLE)

main.o: main.cu params.h structs.h
	$(CC) $(FLAGS) $(CFLAGS) $(COMPUTE_CAPABILITY_FLAGS) main.cu 

kernel.o: kernel.cu params.h structs.h kernel.h
	$(CC) $(FLAGS) $(CFLAGS) $(COMPUTE_CAPABILITY_FLAGS) kernel.cu 

$(EXECUTABLE): $(OBJECTS) $(CUDAOBJECTS)
	$(CC) $(FLAGS) $^ -o $@


###################################
#target for creating shared library for Python

#These parameters should be changed as a function of the machine
#NUMGPU- the number of GPUs (default=1), 
#NTHREADSCPU- the number of physical cores on the platform (default=16)
PARAMSFORSYSTEM=\
-DNUMGPU=1\
-DNTHREADSCPU=16\




#Default parameter set; parameters achieve the best performance, and return the periods and pgram
PARAMSDEFAULTS=\
-DBLOCKSIZE=512\
-DPINNED=1\
-DRETURNPGRAM=1\
-DSIZEPINNEDBUFFERMIB=8\
-DSHMEM=0\
-DNSTREAMS=3\
-DPRINTPERIODS=0\
-DPRINTPGRAM=0\
-DALPHA=0.35\
-DNORMALIZEPGRAM=1\




#make the noerror/error and float/double targets (4 total)
make_python_shared_libs: make_python_noerror_float make_python_noerror_double make_python_error_float make_python_error_double

make_python_noerror_float: main.cu structs.h 
	$(CC) -Xcompiler -fPIC --shared $(PARAMSFORSYSTEM) $(PARAMSDEFAULTS) -DPYTHON -DERROR=0 -DDTYPE=float $(FLAGS) $(CUDAARCHFLAGS) -c main.cu 
	$(CC) -Xcompiler -fPIC --shared $(PARAMSFORSYSTEM) $(PARAMSDEFAULTS) -DPYTHON -DERROR=0 -DDTYPE=float $(FLAGS) $(CUDAARCHFLAGS) -c kernel.cu 
	$(CC) -Xcompiler -fPIC --shared $(PARAMSFORSYSTEM) $(PARAMSDEFAULTS) -DPYTHON -DERROR=0 -DDTYPE=float $(FLAGS) $(CUDAARCHFLAGS) main.o kernel.o -o libpylsnoerrorfloat.so

make_python_noerror_double: main.cu structs.h 
	$(CC) -Xcompiler -fPIC --shared $(PARAMSFORSYSTEM) $(PARAMSDEFAULTS) -DPYTHON -DERROR=0 -DDTYPE=double $(FLAGS) $(CUDAARCHFLAGS) -c main.cu 
	$(CC) -Xcompiler -fPIC --shared $(PARAMSFORSYSTEM) $(PARAMSDEFAULTS) -DPYTHON -DERROR=0 -DDTYPE=double $(FLAGS) $(CUDAARCHFLAGS) -c kernel.cu 
	$(CC) -Xcompiler -fPIC --shared $(PARAMSFORSYSTEM) $(PARAMSDEFAULTS) -DPYTHON -DERROR=0 -DDTYPE=double $(FLAGS) $(CUDAARCHFLAGS) main.o kernel.o -o libpylsnoerrordouble.so

make_python_error_float: main.cu structs.h 
	$(CC) -Xcompiler -fPIC --shared $(PARAMSFORSYSTEM) $(PARAMSDEFAULTS) -DPYTHON -DERROR=1 -DDTYPE=float $(FLAGS) $(CUDAARCHFLAGS) -c main.cu 
	$(CC) -Xcompiler -fPIC --shared $(PARAMSFORSYSTEM) $(PARAMSDEFAULTS) -DPYTHON -DERROR=1 -DDTYPE=float $(FLAGS) $(CUDAARCHFLAGS) -c kernel.cu 
	$(CC) -Xcompiler -fPIC --shared $(PARAMSFORSYSTEM) $(PARAMSDEFAULTS) -DPYTHON -DERROR=1 -DDTYPE=float $(FLAGS) $(CUDAARCHFLAGS) main.o kernel.o -o libpylserrorfloat.so

make_python_error_double: main.cu structs.h 
	$(CC) -Xcompiler -fPIC --shared $(PARAMSFORSYSTEM) $(PARAMSDEFAULTS) -DPYTHON -DERROR=1 -DDTYPE=double $(FLAGS) $(CUDAARCHFLAGS) -c main.cu 
	$(CC) -Xcompiler -fPIC --shared $(PARAMSFORSYSTEM) $(PARAMSDEFAULTS) -DPYTHON -DERROR=1 -DDTYPE=double $(FLAGS) $(CUDAARCHFLAGS) -c kernel.cu 
	$(CC) -Xcompiler -fPIC --shared $(PARAMSFORSYSTEM) $(PARAMSDEFAULTS) -DPYTHON -DERROR=1 -DDTYPE=double $(FLAGS) $(CUDAARCHFLAGS) main.o kernel.o -o libpylserrordouble.so

	

###################################


clean:
	rm $(CUDAOBJECTS) $(OBJECTS)
	rm main libpylsnoerrorfloat.so libpylsnoerrordouble.so libpylserrorfloat.so libpylserrordouble.so




