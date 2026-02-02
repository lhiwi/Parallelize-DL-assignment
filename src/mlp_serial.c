#include "nn_utils.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/*
Serial baseline deep learning training in C.

Model: 2-layer MLP for binary classification (2 logits + softmax)
Input: D=1024 features (32x32 grayscale flattened)
Hidden: H units with ReLU
Output: 2 logits -> softmax -> class probabilities

Training: mini-batch SGD
We measure:
  - loss per epoch
  - train accuracy per epoch
  - test accuracy per epoch
  - time per epoch + total time

This is the "serial baseline" required by the assignment.
*/

// ------------------------ Small helpers ------------------------

// Simple deterministic RNG (so runs are reproducible)
static unsigned long long g_seed = 42ULL;

static void set_seed_local(unsigned long long s) {
    g_seed = (s == 0 ? 1ULL : s);
}

static float rand_uniform(float a, float b) {
    // Linear congruential generator (LCG)
    g_seed = 6364136223846793005ULL * g_seed + 1ULL;
    // Take top bits for a float in [0,1)
    unsigned int r = (unsigned int)(g_seed >> 32);
    float u = (float)r / (float)0xFFFFFFFFu;
    return a + (b - a) * u;
}

static float dot(const float *a, const float *b, int n) {
    float s = 0.0f;
    for (int i = 0; i < n; i++) s += a[i] * b[i];
    return s;
}

/*
Matrix-vector multiply:
W is row-major shape [out][in]
x shape [in]
y shape [out]
*/
static void matvec(const float *W, const float *x, float *y, int out, int in) {
    for (int o = 0; o < out; o++) {
        y[o] = dot(&W[o * in], x, in);
    }
}

static void relu_forward(const float *z, float *a, int n) {
    for (int i = 0; i < n; i++) a[i] = (z[i] > 0.0f) ? z[i] : 0.0f;
}

static void softmax2(float z0, float z1, float *p0, float *p1) {
    // stable 2-class softmax
    float m = (z0 > z1) ? z0 : z1;
    float e0 = expf(z0 - m);
    float e1 = expf(z1 - m);
    float s = e0 + e1;
    *p0 = e0 / s;
    *p1 = e1 / s;
}

static int argmax2(float a, float b) {
    return (a >= b) ? 0 : 1;
}

static float cross_entropy_2(float p0, float p1, int y) {
    // L = -log(p_y)
    const float eps = 1e-8f;
    float py = (y == 0) ? p0 : p1;
    if (py < eps) py = eps;
    return -logf(py);
}

static void init_weights(float *W, int n, float scale) {
    for (int i = 0; i < n; i++) W[i] = rand_uniform(-scale, scale);
}

// ------------------------ Evaluation ------------------------

static float evaluate_accuracy(const Dataset *ds,
                               const float *W1, const float *b1,
                               const float *W2, const float *b2,
                               int H) {
    int N = ds->N;
    int D = ds->D;
    int correct = 0;

    float *z1 = (float*)malloc((size_t)H * sizeof(float));
    float *a1 = (float*)malloc((size_t)H * sizeof(float));
    if (!z1 || !a1) { fprintf(stderr, "OOM in eval\n"); exit(1); }

    for (int i = 0; i < N; i++) {
        const float *x = &ds->X[(size_t)i * (size_t)D];
        int y = (int)ds->y[i];

        // forward
        matvec(W1, x, z1, H, D);
        for (int h = 0; h < H; h++) z1[h] += b1[h];
        relu_forward(z1, a1, H);

        float z20 = dot(&W2[0 * H], a1, H) + b2[0];
        float z21 = dot(&W2[1 * H], a1, H) + b2[1];

        float p0, p1;
        softmax2(z20, z21, &p0, &p1);

        int pred = argmax2(p0, p1);
        if (pred == y) correct++;
    }

    free(z1);
    free(a1);
    return (float)correct / (float)N;
}

// ------------------------ CLI ------------------------

static void usage(const char *prog) {
    printf("Usage: %s -train data/train.bin -test data/test.bin [options]\n", prog);
    printf("Options:\n");
    printf("  -epochs E   (default 5)\n");
    printf("  -batch B    (default 64)\n");
    printf("  -lr LR      (default 0.1)\n");
    printf("  -hidden H   (default 128)\n");
    printf("  -seed S     (default 42)\n");
    printf("  -log FILE   (default results/serial_log.csv)\n");
}

int main(int argc, char **argv) {
    const char *train_path = NULL;
    const char *test_path  = NULL;

    int epochs = 5;
    int batch  = 64;
    float lr   = 0.1f;
    int H      = 128;
    unsigned long long seed = 42ULL;
    const char *log_path = "results/serial_log.csv";

    // Parse args
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "-train") && i + 1 < argc) train_path = argv[++i];
        else if (!strcmp(argv[i], "-test") && i + 1 < argc) test_path = argv[++i];
        else if (!strcmp(argv[i], "-epochs") && i + 1 < argc) epochs = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-batch") && i + 1 < argc) batch = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-lr") && i + 1 < argc) lr = (float)atof(argv[++i]);
        else if (!strcmp(argv[i], "-hidden") && i + 1 < argc) H = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-seed") && i + 1 < argc) seed = (unsigned long long)atoll(argv[++i]);
        else if (!strcmp(argv[i], "-log") && i + 1 < argc) log_path = argv[++i];
        else { usage(argv[0]); return 1; }
    }

    if (!train_path || !test_path) { usage(argv[0]); return 1; }
    if (epochs <= 0 || batch <= 0 || H <= 0) { fprintf(stderr, "Bad hyperparameters\n"); return 1; }

    // Load datasets
    Dataset train, test;
    if (load_bin_dataset(train_path, &train) != 0) { fprintf(stderr, "Failed to load train\n"); return 1; }
    if (load_bin_dataset(test_path, &test) != 0)  { fprintf(stderr, "Failed to load test\n");  return 1; }
    if (train.D != test.D) { fprintf(stderr, "Train/Test D mismatch\n"); return 1; }

    int N = train.N;
    int D = train.D;

    // Initialize model parameters
    set_seed_local(seed);

    // W1: [H][D], b1: [H]
    // W2: [2][H], b2: [2]
    float *W1 = (float*)malloc((size_t)H * (size_t)D * sizeof(float));
    float *b1 = (float*)calloc((size_t)H, sizeof(float));
    float *W2 = (float*)malloc((size_t)2 * (size_t)H * sizeof(float));
    float *b2 = (float*)calloc(2, sizeof(float));

    // Gradients (batch-accumulated)
    float *dW1 = (float*)malloc((size_t)H * (size_t)D * sizeof(float));
    float *db1 = (float*)malloc((size_t)H * sizeof(float));
    float *dW2 = (float*)malloc((size_t)2 * (size_t)H * sizeof(float));
    float  db2[2];

    if (!W1 || !b1 || !W2 || !b2 || !dW1 || !db1 || !dW2) {
        fprintf(stderr, "OOM allocating parameters\n");
        return 1;
    }

    init_weights(W1, H * D, 0.05f);
    init_weights(W2, 2 * H, 0.05f);

    // Temporary buffers for one sample (forward/backward)
    float *z1  = (float*)malloc((size_t)H * sizeof(float));
    float *a1  = (float*)malloc((size_t)H * sizeof(float));
    float *da1 = (float*)malloc((size_t)H * sizeof(float));
    float *dz1 = (float*)malloc((size_t)H * sizeof(float));
    if (!z1 || !a1 || !da1 || !dz1) { fprintf(stderr, "OOM temp buffers\n"); return 1; }

    // Ensure results directory exists
    (void)system("mkdir -p results");

    // Open CSV log
    FILE *logf = fopen(log_path, "w");
    if (!logf) { perror("fopen log"); return 1; }
    fprintf(logf, "epoch,epoch_time_s,train_loss,train_acc,test_acc\n");

    printf("Serial baseline (MLP): N=%d D=%d H=%d epochs=%d batch=%d lr=%.4f\n",
           N, D, H, epochs, batch, lr);

    double t_total0 = now_seconds();

    // ------------------------ TRAINING LOOP ------------------------
    for (int e = 1; e <= epochs; e++) {
        double t0 = now_seconds();

        double loss_sum = 0.0;
        int correct = 0;

        // Loop over mini-batches
        for (int start = 0; start < N; start += batch) {
            int bs = batch;
            if (start + bs > N) bs = N - start;

            // Zero gradients for this batch
            memset(dW1, 0, (size_t)H * (size_t)D * sizeof(float));
            memset(db1, 0, (size_t)H * sizeof(float));
            memset(dW2, 0, (size_t)2 * (size_t)H * sizeof(float));
            db2[0] = 0.0f; db2[1] = 0.0f;

            // Accumulate gradients across samples in this batch
            for (int ii = 0; ii < bs; ii++) {
                int i = start + ii;
                const float *x = &train.X[(size_t)i * (size_t)D];
                int y = (int)train.y[i];

                // ---------- Forward pass ----------
                matvec(W1, x, z1, H, D);
                for (int h = 0; h < H; h++) z1[h] += b1[h];
                relu_forward(z1, a1, H);

                float z20 = dot(&W2[0 * H], a1, H) + b2[0];
                float z21 = dot(&W2[1 * H], a1, H) + b2[1];

                float p0, p1;
                softmax2(z20, z21, &p0, &p1);

                // Loss + accuracy bookkeeping
                loss_sum += (double)cross_entropy_2(p0, p1, y);
                int pred = argmax2(p0, p1);
                if (pred == y) correct++;

                // ---------- Backward pass ----------
                // For softmax + cross-entropy:
                // dL/dz2 = p - onehot(y)
                float dz20 = p0 - (y == 0 ? 1.0f : 0.0f);
                float dz21 = p1 - (y == 1 ? 1.0f : 0.0f);

                // Gradients for W2 and b2
                for (int h = 0; h < H; h++) {
                    dW2[0 * H + h] += dz20 * a1[h];
                    dW2[1 * H + h] += dz21 * a1[h];
                }
                db2[0] += dz20;
                db2[1] += dz21;

                // da1 = W2^T * dz2
                for (int h = 0; h < H; h++) {
                    da1[h] = W2[0 * H + h] * dz20 + W2[1 * H + h] * dz21;
                }

                // dz1 = da1 * ReLU'(z1)
                for (int h = 0; h < H; h++) {
                    dz1[h] = (z1[h] > 0.0f) ? da1[h] : 0.0f;
                }

                // Gradients for W1 and b1
                for (int h = 0; h < H; h++) {
                    float g = dz1[h];
                    db1[h] += g;

                    float *row = &dW1[(size_t)h * (size_t)D];
                    for (int d = 0; d < D; d++) {
                        row[d] += g * x[d];
                    }
                }
            }

            // ---------- Parameter update (SGD) ----------
            // We average the gradients over the batch (divide by bs)
            float inv_bs = 1.0f / (float)bs;

            for (int h = 0; h < H; h++) {
                b1[h] -= lr * db1[h] * inv_bs;

                float *wrow = &W1[(size_t)h * (size_t)D];
                float *grow = &dW1[(size_t)h * (size_t)D];
                for (int d = 0; d < D; d++) {
                    wrow[d] -= lr * grow[d] * inv_bs;
                }
            }

            for (int h = 0; h < H; h++) {
                W2[0 * H + h] -= lr * dW2[0 * H + h] * inv_bs;
                W2[1 * H + h] -= lr * dW2[1 * H + h] * inv_bs;
            }
            b2[0] -= lr * db2[0] * inv_bs;
            b2[1] -= lr * db2[1] * inv_bs;
        }

        double t1 = now_seconds();
        double epoch_time = t1 - t0;

        float train_loss = (float)(loss_sum / (double)N);
        float train_acc  = (float)correct / (float)N;
        float test_acc   = evaluate_accuracy(&test, W1, b1, W2, b2, H);

        printf("Epoch %d: time=%.3fs loss=%.4f train_acc=%.4f test_acc=%.4f\n",
               e, epoch_time, train_loss, train_acc, test_acc);

        fprintf(logf, "%d,%.6f,%.6f,%.6f,%.6f\n",
                e, epoch_time, train_loss, train_acc, test_acc);
        fflush(logf);
    }

    double t_total1 = now_seconds();
    printf("Total training time: %.3fs\n", t_total1 - t_total0);

    fclose(logf);

    // Cleanup
    free(z1); free(a1); free(da1); free(dz1);
    free(W1); free(b1); free(W2); free(b2);
    free(dW1); free(db1); free(dW2);
    free_dataset(&train);
    free_dataset(&test);

    return 0;
}
