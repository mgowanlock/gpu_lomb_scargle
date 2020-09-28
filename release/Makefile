

SOURCES = main.cu 
CUDAOBJECTS = main.o kernel.o
CC = nvcc
EXECUTABLE = main
COMPUTE_CAPABILITY = -arch=compute_75 -code=sm_75


FLAGS = -std=c++11 -O3 -Xcompiler -fopenmp  -lcuda -lineinfo -g
CFLAGS = -c -D_MWAITXINTRIN_H_INCLUDED -D_FORCE_INLINES


all: $(EXECUTABLE)




main.o: main.cu structs.h
	$(CC) $(FLAGS) $(CFLAGS) $(COMPUTE_CAPABILITY) main.cu 

kernel.o: kernel.cu structs.h kernel.h
	$(CC) $(FLAGS) $(CFLAGS) $(COMPUTE_CAPABILITY) kernel.cu 

$(EXECUTABLE): $(OBJECTS) $(CUDAOBJECTS)
	$(CC) $(FLAGS) $^ -o $@




clean:
	rm $(CUDAOBJECTS)
	rm main




