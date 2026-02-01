#ifndef NN_UTILS_H
#define NN_UTILS_H

#include <stdint.h>

// A simple struct to hold our dataset in memory.
typedef struct {
    int N;      // number of samples
    int D;      // number of features per sample (should be 1024)
    float *X;   // flattened features, size N*D
    uint8_t *y; // labels, size N (0 or 1)
} Dataset;

// Loads our custom binary dataset format.
// File format:
//   int32 N
//   int32 D
//   float32 X[N*D]  (row-major)
//   uint8  y[N]
int load_bin_dataset(const char *path, Dataset *ds);

// Frees memory allocated by load_bin_dataset.
void free_dataset(Dataset *ds);

// Returns a high-resolution wall-clock time in seconds.
// Used for timing epochs and total runtime.
double now_seconds(void);

#endif
