
SOURCES = main.cu 
CUDAOBJECTS = main.o
CC = nvcc
EXECUTABLE = main

FLAGS =  -std=c++14 -O3 -Xcompiler -fopenmp -arch=compute_60 -code=sm_60 -lcuda -lineinfo
CFLAGS = -c -D_MWAITXINTRIN_H_INCLUDED -D_FORCE_INLINES


all: $(EXECUTABLE)

main.o: main.cu kernel.h
	$(CC) $(FLAGS) $(CFLAGS) main.cu 

$(EXECUTABLE): $(OBJECTS) $(CUDAOBJECTS)
	$(CC) $(FLAGS) $^ -o $@

clean:
	rm $(CUDAOBJECTS)
	rm main




