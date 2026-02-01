CC=gcc
CFLAGS=-O3 -march=native -Wall -Wextra -std=c11
LDFLAGS=-lm

SRC=src
BIN=bin

OMPFLAGS=-fopenmp

all: dirs mlp_serial cnn_serial cnn_omp

dirs:
	mkdir -p $(BIN)

mlp_serial: dirs
	$(CC) $(CFLAGS) -o $(BIN)/mlp_serial $(SRC)/mlp_serial.c $(SRC)/nn_utils.c $(LDFLAGS)

cnn_serial: dirs
	$(CC) $(CFLAGS) -o $(BIN)/cnn_serial $(SRC)/cnn_serial.c $(SRC)/nn_utils.c $(LDFLAGS)

cnn_omp: dirs
	$(CC) $(CFLAGS) $(OMPFLAGS) -o $(BIN)/cnn_omp $(SRC)/cnn_omp.c $(SRC)/nn_utils.c $(LDFLAGS)

clean:
	rm -rf $(BIN) *.o

.PHONY: all dirs mlp_serial cnn_serial cnn_omp clean

cnn_hybrid: dirs
	mpicc -O3 -march=native -Wall -Wextra -std=c11 -fopenmp -o bin/cnn_hybrid src/cnn_hybrid.c src/nn_utils.c -lm
