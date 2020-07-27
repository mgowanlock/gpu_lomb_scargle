
#see params.h for the parameters




SOURCES = main.cu 
CUDAOBJECTS = main.o kernel.o
CC = nvcc
EXECUTABLE = main
COMPUTE_CAPABILITY= -arch=compute_60 -code=sm_60


FLAGS =  -std=c++11 -O3 -Xcompiler -fopenmp  -lcuda -lineinfo 
CFLAGS = -c -D_MWAITXINTRIN_H_INCLUDED -D_FORCE_INLINES


all: $(EXECUTABLE)




main.o: main.cu params.h structs.h
	$(CC) $(FLAGS) $(CFLAGS) $(COMPUTE_CAPABILITY) main.cu 

kernel.o: kernel.cu params.h structs.h kernel.h
	$(CC) $(FLAGS) $(CFLAGS) $(COMPUTE_CAPABILITY) kernel.cu 

$(EXECUTABLE): $(OBJECTS) $(CUDAOBJECTS)
	$(CC) $(FLAGS) $^ -o $@




clean:
	rm $(CUDAOBJECTS)
	rm main




