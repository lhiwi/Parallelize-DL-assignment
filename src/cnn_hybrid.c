// src/cnn_hybrid.c
#include "nn_utils.h"
#include <mpi.h>
#include <omp.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

/*
Hybrid CNN training: MPI (ranks) + OpenMP (threads)

Design (data-parallel, batch-synchronous SGD):
- Each mini-batch has global size bs.
- MPI ranks split the samples of the batch: rank r computes ii in [lo, hi).
- Within each rank, OpenMP parallelizes over that local sample range.
- Each rank produces local gradients (sum over its samples).
- MPI_Allreduce sums gradients across ranks -> global gradient.
- All ranks apply the SAME update once per batch (matches serial semantics).

We also parallelize evaluation:
- MPI ranks split the test set; OpenMP threads split within rank.
- MPI_Allreduce sums correct predictions to get global test accuracy.
*/

// ---------- deterministic RNG for init (rank 0 only, then broadcast) ----------
static unsigned long long g_seed = 42ULL;
static void set_seed_local(unsigned long long s){ g_seed = (s ? s : 1ULL); }
static float rand_uniform(float a, float b){
    g_seed = 6364136223846793005ULL * g_seed + 1ULL;
    unsigned int r = (unsigned int)(g_seed >> 32);
    float u = (float)r / (float)0xFFFFFFFFu;
    return a + (b - a) * u;
}

// ---------- tiny helpers ----------
static void softmax2(float z0, float z1, float *p0, float *p1){
    float m = (z0 > z1) ? z0 : z1;
    float e0 = expf(z0 - m);
    float e1 = expf(z1 - m);
    float s = e0 + e1;
    *p0 = e0 / s;
    *p1 = e1 / s;
}
static int argmax2(float a, float b){ return (a >= b) ? 0 : 1; }
static float cross_entropy_2(float p0, float p1, int y){
    const float eps = 1e-8f;
    float py = (y==0) ? p0 : p1;
    if(py < eps) py = eps;
    return -logf(py);
}
static inline int idx3(int c, int i, int j, int H, int W){
    return (c * H + i) * W + j;
}

// ---------- ops: conv/relu/pool/fc ----------
static void conv3x3_pad1_forward(
    const float *in, int C, int H, int W,
    const float *Wf, const float *bf,
    float *out, int F
){
    for(int f=0; f<F; f++){
        for(int i=0; i<H; i++){
            for(int j=0; j<W; j++){
                float s = bf[f];
                const float *wf = &Wf[ ((f*C)*3)*3 ];
                for(int c=0; c<C; c++){
                    const float *wfc = &wf[(c*3)*3];
                    for(int kh=0; kh<3; kh++){
                        int ii = i + kh - 1;
                        if(ii < 0 || ii >= H) continue;
                        for(int kw=0; kw<3; kw++){
                            int jj = j + kw - 1;
                            if(jj < 0 || jj >= W) continue;
                            s += wfc[kh*3 + kw] * in[idx3(c, ii, jj, H, W)];
                        }
                    }
                }
                out[idx3(f, i, j, H, W)] = s;
            }
        }
    }
}
static void relu_forward(const float *x, float *y, int n){
    for(int i=0;i<n;i++) y[i] = (x[i] > 0.0f) ? x[i] : 0.0f;
}
static void relu_backward(const float *preact, const float *dout, float *din, int n){
    for(int i=0;i<n;i++) din[i] = (preact[i] > 0.0f) ? dout[i] : 0.0f;
}

static void maxpool2x2_forward(const float *in, int C, int H, int W,
                               float *out, uint8_t *idx){
    int OH = H/2, OW = W/2;
    for(int c=0;c<C;c++){
        for(int i=0;i<OH;i++){
            for(int j=0;j<OW;j++){
                int ii = i*2, jj = j*2;
                float a = in[idx3(c, ii,   jj,   H, W)];
                float b = in[idx3(c, ii,   jj+1, H, W)];
                float d = in[idx3(c, ii+1, jj,   H, W)];
                float e = in[idx3(c, ii+1, jj+1, H, W)];
                float m = a; uint8_t k = 0;
                if(b > m){ m=b; k=1; }
                if(d > m){ m=d; k=2; }
                if(e > m){ m=e; k=3; }
                out[idx3(c, i, j, OH, OW)] = m;
                idx[idx3(c, i, j, OH, OW)] = k;
            }
        }
    }
}

static void maxpool2x2_backward(const float *dout, const uint8_t *idx,
                                int C, int H, int W, float *din){
    int OH = H/2, OW = W/2;
    memset(din, 0, (size_t)C*(size_t)H*(size_t)W*sizeof(float));
    for(int c=0;c<C;c++){
        for(int i=0;i<OH;i++){
            for(int j=0;j<OW;j++){
                float g = dout[idx3(c, i, j, OH, OW)];
                uint8_t k = idx[idx3(c, i, j, OH, OW)];
                int ii = i*2, jj = j*2;
                int di = (k==2 || k==3) ? 1 : 0;
                int dj = (k==1 || k==3) ? 1 : 0;
                din[idx3(c, ii+di, jj+dj, H, W)] += g;
            }
        }
    }
}

static void fc_forward(const float *W, const float *b, const float *x, float *z){
    float s0 = b[0], s1 = b[1];
    const float *w0 = &W[0*1024];
    const float *w1 = &W[1*1024];
    for(int i=0;i<1024;i++){
        float xi = x[i];
        s0 += w0[i]*xi;
        s1 += w1[i]*xi;
    }
    z[0]=s0; z[1]=s1;
}

static void fc_backward(const float *W, const float *x, const float *dz,
                        float *dW, float *db, float *dx){
    for(int i=0;i<1024;i++){
        dW[0*1024 + i] += dz[0] * x[i];
        dW[1*1024 + i] += dz[1] * x[i];
        dx[i] = W[0*1024 + i]*dz[0] + W[1*1024 + i]*dz[1];
    }
    db[0] += dz[0];
    db[1] += dz[1];
}

static void conv3x3_pad1_backward(
    const float *in, int C, int H, int W,
    const float *Wf, int F,
    const float *dout,
    float *dWf, float *dbf,
    float *din
){
    memset(din, 0, (size_t)C*(size_t)H*(size_t)W*sizeof(float));

    for(int f=0; f<F; f++){
        const float *wf = &Wf[ ((f*C)*3)*3 ];
        float *dwf = &dWf[ ((f*C)*3)*3 ];

        for(int i=0; i<H; i++){
            for(int j=0; j<W; j++){
                float g = dout[idx3(f, i, j, H, W)];
                dbf[f] += g;

                for(int c=0; c<C; c++){
                    const float *wfc = &wf[(c*3)*3];
                    float *dwfc = &dwf[(c*3)*3];

                    for(int kh=0; kh<3; kh++){
                        int ii = i + kh - 1;
                        if(ii < 0 || ii >= H) continue;
                        for(int kw=0; kw<3; kw++){
                            int jj = j + kw - 1;
                            if(jj < 0 || jj >= W) continue;

                            float x = in[idx3(c, ii, jj, H, W)];
                            dwfc[kh*3 + kw] += g * x;
                            din[idx3(c, ii, jj, H, W)] += wfc[kh*3 + kw] * g;
                        }
                    }
                }
            }
        }
    }
}

// ---------- eval: MPI ranks split test set; OpenMP within rank ----------
static float evaluate_accuracy_mpi(const Dataset *ds,
                                  const float *Wc1, const float *bc1,
                                  const float *Wc2, const float *bc2,
                                  const float *Wfc, const float *bfc,
                                  MPI_Comm comm)
{
    int rank=0, nprocs=1;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &nprocs);

    int N = ds->N;
    int lo = (int)((long long)rank * N / nprocs);
    int hi = (int)((long long)(rank+1) * N / nprocs);

    int local_correct = 0;

#pragma omp parallel for reduction(+:local_correct) schedule(static)
    for(int n=lo; n<hi; n++){
        float conv1[8*32*32], act1[8*32*32], pool1[8*16*16];
        uint8_t p1idx[8*16*16];
        float conv2[16*16*16], act2[16*16*16], pool2[16*8*8];
        uint8_t p2idx[16*8*8];
        float logits[2];

        const float *x = &ds->X[(size_t)n*1024];

        conv3x3_pad1_forward(x, 1, 32, 32, Wc1, bc1, conv1, 8);
        relu_forward(conv1, act1, 8*32*32);
        maxpool2x2_forward(act1, 8, 32, 32, pool1, p1idx);

        conv3x3_pad1_forward(pool1, 8, 16, 16, Wc2, bc2, conv2, 16);
        relu_forward(conv2, act2, 16*16*16);
        maxpool2x2_forward(act2, 16, 16, 16, pool2, p2idx);

        fc_forward(Wfc, bfc, pool2, logits);
        float p0,p1; softmax2(logits[0], logits[1], &p0, &p1);
        int pred = argmax2(p0,p1);
        if(pred == (int)ds->y[n]) local_correct++;
    }

    int global_correct = 0;
    MPI_Allreduce(&local_correct, &global_correct, 1, MPI_INT, MPI_SUM, comm);

    return (float)global_correct / (float)N;
}

static void usage(const char *p){
    printf("Usage: %s -train data/train.bin -test data/test.bin [options]\n", p);
    printf("Options:\n");
    printf("  -epochs E   (default 5)\n");
    printf("  -batch B    (default 32)\n");
    printf("  -lr LR      (default 0.05)\n");
    printf("  -seed S     (default 42)\n");
    printf("  -log FILE   (default results/cnn_hybrid_log.csv)\n");
}

int main(int argc, char **argv){
    MPI_Init(&argc, &argv);

    MPI_Comm comm = MPI_COMM_WORLD;
    int rank=0, nprocs=1;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &nprocs);

    const char *train_path=NULL, *test_path=NULL;
    int epochs=5, batch=32;
    float lr=0.05f;
    unsigned long long seed=42ULL;
    const char *log_path="results/cnn_hybrid_log.csv";

    for(int i=1;i<argc;i++){
        if(!strcmp(argv[i],"-train") && i+1<argc) train_path=argv[++i];
        else if(!strcmp(argv[i],"-test") && i+1<argc) test_path=argv[++i];
        else if(!strcmp(argv[i],"-epochs") && i+1<argc) epochs=atoi(argv[++i]);
        else if(!strcmp(argv[i],"-batch") && i+1<argc) batch=atoi(argv[++i]);
        else if(!strcmp(argv[i],"-lr") && i+1<argc) lr=(float)atof(argv[++i]);
        else if(!strcmp(argv[i],"-seed") && i+1<argc) seed=(unsigned long long)atoll(argv[++i]);
        else if(!strcmp(argv[i],"-log") && i+1<argc) log_path=argv[++i];
        else { if(rank==0) usage(argv[0]); MPI_Finalize(); return 1; }
    }
    if(!train_path || !test_path){ if(rank==0) usage(argv[0]); MPI_Finalize(); return 1; }

    Dataset train, test;
    if(load_bin_dataset(train_path,&train)!=0){ if(rank==0) fprintf(stderr,"Failed to load train\n"); MPI_Finalize(); return 1; }
    if(load_bin_dataset(test_path,&test)!=0){ if(rank==0) fprintf(stderr,"Failed to load test\n"); free_dataset(&train); MPI_Finalize(); return 1; }
    if(train.D!=1024 || test.D!=1024){ if(rank==0) fprintf(stderr,"Expected D=1024\n"); free_dataset(&train); free_dataset(&test); MPI_Finalize(); return 1; }

    int N = train.N;
    int T = omp_get_max_threads();

    // Parameters
    float Wc1[8*1*3*3], bc1[8];
    float Wc2[16*8*3*3], bc2[16];
    float Wfc[2*1024], bfc[2];

    if(rank==0){
        set_seed_local(seed);
        for(int i=0;i<8*1*3*3;i++) Wc1[i]=rand_uniform(-0.05f,0.05f);
        for(int i=0;i<16*8*3*3;i++) Wc2[i]=rand_uniform(-0.05f,0.05f);
        for(int i=0;i<2*1024;i++) Wfc[i]=rand_uniform(-0.05f,0.05f);
        memset(bc1,0,sizeof(bc1));
        memset(bc2,0,sizeof(bc2));
        memset(bfc,0,sizeof(bfc));
    }

    // Broadcast weights so all ranks start identical
    MPI_Bcast(Wc1, 8*1*3*3, MPI_FLOAT, 0, comm);
    MPI_Bcast(bc1, 8,         MPI_FLOAT, 0, comm);
    MPI_Bcast(Wc2, 16*8*3*3,  MPI_FLOAT, 0, comm);
    MPI_Bcast(bc2, 16,        MPI_FLOAT, 0, comm);
    MPI_Bcast(Wfc, 2*1024,    MPI_FLOAT, 0, comm);
    MPI_Bcast(bfc, 2,         MPI_FLOAT, 0, comm);

    (void)system("mkdir -p results");

    FILE *logf = NULL;
    if(rank==0){
        logf = fopen(log_path, "w");
        if(!logf){ perror("fopen log"); free_dataset(&train); free_dataset(&test); MPI_Finalize(); return 1; }
        fprintf(logf,"epoch,epoch_time_s,train_loss,train_acc,test_acc,np,threads\n");
    }

    if(rank==0){
        printf("Hybrid CNN (MPI+OpenMP): N=%d D=%d epochs=%d batch=%d lr=%.4f np=%d threads=%d\n",
               N, train.D, epochs, batch, lr, nprocs, T);
    }

    // Gradient packing: total 3298 floats
    enum { G_WC1=72, G_BC1=8, G_WC2=1152, G_BC2=16, G_WFC=2048, G_BFC=2, G_TOTAL=3298 };

    float *g_local  = (float*)malloc(G_TOTAL*sizeof(float));
    float *g_global = (float*)malloc(G_TOTAL*sizeof(float));
    if(!g_local || !g_global){
        if(rank==0) fprintf(stderr,"OOM gradient buffers\n");
        free_dataset(&train); free_dataset(&test);
        MPI_Finalize(); return 1;
    }

    // Per-thread local grads inside rank
    float *g_thr = (float*)malloc((size_t)T * (size_t)G_TOTAL * sizeof(float));
    if(!g_thr){
        if(rank==0) fprintf(stderr,"OOM thread gradient buffers\n");
        free(g_local); free(g_global);
        free_dataset(&train); free_dataset(&test);
        MPI_Finalize(); return 1;
    }

    double t_total0 = now_seconds();

    for(int e=1; e<=epochs; e++){
        double t0 = now_seconds();

        double epoch_loss_local = 0.0;
        long long epoch_correct_local = 0;

        for(int start=0; start<N; start+=batch){
            int bs = batch;
            if(start+bs > N) bs = N-start;

            // Rank split of batch indices [0, bs)
            int lo = (int)((long long)rank * bs / nprocs);
            int hi = (int)((long long)(rank+1) * bs / nprocs);

            // Zero thread grads
            memset(g_thr, 0, (size_t)T*(size_t)G_TOTAL*sizeof(float));

            double batch_loss_local = 0.0;
            int batch_correct_local = 0;

#pragma omp parallel
            {
                int tid = omp_get_thread_num();
                float *gt = &g_thr[(size_t)tid * (size_t)G_TOTAL];

                float *dWc1 = gt;                 // 72
                float *dbc1 = dWc1 + G_WC1;        // 8
                float *dWc2 = dbc1 + G_BC1;        // 1152
                float *dbc2 = dWc2 + G_WC2;        // 16
                float *dWfc = dbc2 + G_BC2;        // 2048
                float *dbfc = dWfc + G_WFC;        // 2

                // private buffers
                float conv1[8*32*32], act1[8*32*32], pool1[8*16*16];
                uint8_t p1idx[8*16*16];
                float conv2[16*16*16], act2[16*16*16], pool2[16*8*8];
                uint8_t p2idx[16*8*8];
                float logits[2];

                float dpool2[16*8*8];
                float dact2[16*16*16], dconv2[16*16*16];
                float dx_pool1[8*16*16];
                float dpool1[8*16*16];
                float dact1[8*32*32], dconv1[8*32*32];
                float dtmp_flat[1024];
                float dx_img[1*32*32];

                double local_loss = 0.0;
                int local_correct = 0;

#pragma omp for schedule(static)
                for(int ii=lo; ii<hi; ii++){
                    int n = start + ii;
                    const float *x = &train.X[(size_t)n*1024];
                    int y = (int)train.y[n];

                    // forward
                    conv3x3_pad1_forward(x, 1, 32, 32, Wc1, bc1, conv1, 8);
                    relu_forward(conv1, act1, 8*32*32);
                    maxpool2x2_forward(act1, 8, 32, 32, pool1, p1idx);

                    conv3x3_pad1_forward(pool1, 8, 16, 16, Wc2, bc2, conv2, 16);
                    relu_forward(conv2, act2, 16*16*16);
                    maxpool2x2_forward(act2, 16, 16, 16, pool2, p2idx);

                    fc_forward(Wfc, bfc, pool2, logits);
                    float p0,p1; softmax2(logits[0], logits[1], &p0, &p1);

                    local_loss += (double)cross_entropy_2(p0,p1,y);
                    int pred = argmax2(p0,p1);
                    if(pred==y) local_correct++;

                    // backward softmax+CE
                    float dz[2];
                    dz[0] = p0 - (y==0 ? 1.0f : 0.0f);
                    dz[1] = p1 - (y==1 ? 1.0f : 0.0f);

                    // FC backward
                    fc_backward(Wfc, pool2, dz, dWfc, dbfc, dtmp_flat);
                    memcpy(dpool2, dtmp_flat, sizeof(dpool2));

                    maxpool2x2_backward(dpool2, p2idx, 16, 16, 16, dact2);
                    relu_backward(conv2, dact2, dconv2, 16*16*16);

                    conv3x3_pad1_backward(pool1, 8, 16, 16, Wc2, 16, dconv2, dWc2, dbc2, dx_pool1);
                    memcpy(dpool1, dx_pool1, sizeof(dpool1));

                    maxpool2x2_backward(dpool1, p1idx, 8, 32, 32, dact1);
                    relu_backward(conv1, dact1, dconv1, 8*32*32);

                    conv3x3_pad1_backward(x, 1, 32, 32, Wc1, 8, dconv1, dWc1, dbc1, dx_img);
                }

#pragma omp atomic
                batch_loss_local += local_loss;
#pragma omp atomic
                batch_correct_local += local_correct;
            } // omp parallel

            // Reduce thread grads -> g_local (within rank)
            memset(g_local, 0, G_TOTAL*sizeof(float));
            for(int t=0; t<T; t++){
                float *gt = &g_thr[(size_t)t * (size_t)G_TOTAL];
                for(int i=0;i<G_TOTAL;i++) g_local[i] += gt[i];
            }

            // MPI reduce gradients across ranks
            MPI_Allreduce(g_local, g_global, G_TOTAL, MPI_FLOAT, MPI_SUM, comm);

            // Also get global batch loss/correct for accurate epoch metrics
            double batch_loss_global = 0.0;
            int batch_correct_global = 0;
            MPI_Allreduce(&batch_loss_local, &batch_loss_global, 1, MPI_DOUBLE, MPI_SUM, comm);
            MPI_Allreduce(&batch_correct_local, &batch_correct_global, 1, MPI_INT, MPI_SUM, comm);

            epoch_loss_local += batch_loss_global;     // already global
            epoch_correct_local += batch_correct_global;

            // Apply SGD update on all ranks (same g_global everywhere)
            float inv_bs = 1.0f / (float)bs;

            float *dWc1 = g_global;
            float *dbc1 = dWc1 + G_WC1;
            float *dWc2 = dbc1 + G_BC1;
            float *dbc2 = dWc2 + G_WC2;
            float *dWfc = dbc2 + G_BC2;
            float *dbfc = dWfc + G_WFC;

            for(int i=0;i<G_WC1;i++) Wc1[i] -= lr * dWc1[i] * inv_bs;
            for(int i=0;i<G_BC1;i++) bc1[i] -= lr * dbc1[i] * inv_bs;

            for(int i=0;i<G_WC2;i++) Wc2[i] -= lr * dWc2[i] * inv_bs;
            for(int i=0;i<G_BC2;i++) bc2[i] -= lr * dbc2[i] * inv_bs;

            for(int i=0;i<G_WFC;i++) Wfc[i] -= lr * dWfc[i] * inv_bs;
            for(int i=0;i<G_BFC;i++) bfc[i] -= lr * dbfc[i] * inv_bs;
        }

        double t1 = now_seconds();
        double epoch_time = t1 - t0;

        float train_loss = (float)(epoch_loss_local / (double)N);
        float train_acc  = (float)epoch_correct_local / (float)N;
        float test_acc   = evaluate_accuracy_mpi(&test, Wc1, bc1, Wc2, bc2, Wfc, bfc, comm);

        if(rank==0){
            printf("Epoch %d: time=%.3fs loss=%.4f train_acc=%.4f test_acc=%.4f (np=%d, threads=%d)\n",
                   e, epoch_time, train_loss, train_acc, test_acc, nprocs, T);
            if(logf){
                fprintf(logf, "%d,%.6f,%.6f,%.6f,%.6f,%d,%d\n",
                        e, epoch_time, train_loss, train_acc, test_acc, nprocs, T);
                fflush(logf);
            }
        }
    }

    double t_total1 = now_seconds();
    if(rank==0){
        printf("Total training time: %.3fs\n", t_total1 - t_total0);
    }

    if(logf) fclose(logf);

    free(g_thr);
    free(g_local);
    free(g_global);
    free_dataset(&train);
    free_dataset(&test);

    MPI_Finalize();
    return 0;
}

