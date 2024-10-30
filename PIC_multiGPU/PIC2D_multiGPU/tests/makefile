CC = nvcc
CFLAGS = -ccbin mpic++ -lcudart -gencode=arch=compute_60,code=sm_80 -rdc=true --extended-lambda -Xcompiler "-Wall -O3"
MAIN = main_test_mpi.cu
EXECUTABLE = program

.PHONY: all clean

all: $(EXECUTABLE)

$(EXECUTABLE): $(MAIN)
	$(CC) $(CFLAGS) $(MAIN) -o $(EXECUTABLE)

run: $(EXECUTABLE)
	./$(EXECUTABLE)

clean:
	rm -f $(MAIN) $(EXECUTABLE)
