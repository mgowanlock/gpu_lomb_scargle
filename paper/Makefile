
#see params.h for the parameters




SOURCES = main.cu 
CUDAOBJECTS = main.o kernel.o
CC = nvcc
EXECUTABLE = main

FLAGS =  -std=c++11 -O3 -Xcompiler -fopenmp -arch=compute_60 -code=sm_60 -lcuda -lineinfo 
CFLAGS = -c -D_MWAITXINTRIN_H_INCLUDED -D_FORCE_INLINES


all: $(EXECUTABLE)




main.o: main.cu params.h structs.h
	$(CC) $(FLAGS) $(CFLAGS) main.cu 

kernel.o: kernel.cu params.h structs.h kernel.h
	$(CC) $(FLAGS) $(CFLAGS) kernel.cu 

$(EXECUTABLE): $(OBJECTS) $(CUDAOBJECTS)
	$(CC) $(FLAGS) $^ -o $@




clean:
	rm $(CUDAOBJECTS)
	rm main




