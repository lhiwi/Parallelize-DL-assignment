CC=gcc
CFLAGS=-O3 -march=native -Wall -Wextra -std=c11
LDFLAGS=-lm

SRC=src
BIN=bin

all: dirs serial

dirs:
	mkdir -p $(BIN)

serial: dirs
	$(CC) $(CFLAGS) -o $(BIN)/mlp_serial $(SRC)/mlp_serial.c $(SRC)/nn_utils.c $(LDFLAGS)

clean:
	rm -rf $(BIN) *.o

.PHONY: all dirs serial clean
