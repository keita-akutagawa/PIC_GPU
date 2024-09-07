CC = nvcc
CFLAGS = -ccbin mpic++ -gencode=arch=compute_60,code=sm_80 -rdc=true --extended-lambda -Xcompiler "-Wall -O3"
SRCDIR = ../../PIC1D_multiGPU
SOURCES = $(wildcard $(SRCDIR)/*.cu)
OBJS = $(patsubst $(SRCDIR)/%.cu, $(SRCDIR)/%.o, $(SOURCES))
MAIN = main_two_stream_electron.o
EXECUTABLE = program

.PHONY: all clean

all: $(EXECUTABLE)

$(EXECUTABLE): $(OBJS) $(MAIN)
	$(CC) $(CFLAGS) $(OBJS) $(MAIN) -o $(EXECUTABLE)

$(SRCDIR)/%.o: $(SRCDIR)/%.cu
	$(CC) $(CFLAGS) -c $< -o $@

$(MAIN): main_two_stream_electron.cu
	$(CC) $(CFLAGS) -c $< -o $@

run: $(EXECUTABLE)
	$(EXECUTABLE)

clean:
	rm -f $(OBJS) $(MAIN) $(EXECUTABLE)
