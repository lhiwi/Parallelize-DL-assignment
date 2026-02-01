#define _POSIX_C_SOURCE 199309L
#include "nn_utils.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

// High-resolution wall clock timer
double now_seconds(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

int load_bin_dataset(const char *path, Dataset *ds) {
    // Start with a clean struct (so free_dataset is safe even if we fail early)
    memset(ds, 0, sizeof(*ds));

    FILE *f = fopen(path, "rb");
    if (!f) {
        perror("fopen");
        return -1;
    }

    int N = 0, D = 0;

    // Read header: N and D
    if (fread(&N, sizeof(int), 1, f) != 1) { fclose(f); return -1; }
    if (fread(&D, sizeof(int), 1, f) != 1) { fclose(f); return -1; }

    if (N <= 0 || D <= 0) { fclose(f); return -1; }

    ds->N = N;
    ds->D = D;

    // Allocate memory for X and y
    size_t nX = (size_t)N * (size_t)D;
    ds->X = (float*)malloc(nX * sizeof(float));
    ds->y = (uint8_t*)malloc((size_t)N * sizeof(uint8_t));

    if (!ds->X || !ds->y) {
        fclose(f);
        return -1;
    }

    // Read X and y
    if (fread(ds->X, sizeof(float), nX, f) != nX) { fclose(f); return -1; }
    if (fread(ds->y, sizeof(uint8_t), (size_t)N, f) != (size_t)N) { fclose(f); return -1; }

    fclose(f);
    return 0;
}

void free_dataset(Dataset *ds) {
    free(ds->X);
    free(ds->y);
    ds->X = NULL;
    ds->y = NULL;
    ds->N = 0;
    ds->D = 0;
}
