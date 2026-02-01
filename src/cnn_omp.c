#include "nn_utils.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>

/*
OpenMP parallel CNN training (data parallelism over samples in a mini-batch).

Key design:
- Model parameters (Wc1, bc1, Wc2, bc2, Wfc, bfc) are read-only during the parallel
  region for a given batch.
- Each thread has private activation/backprop buffers and private gradient buffers.
- After processing the batch, we reduce (sum) thread gradients into global gradients
  and apply ONE SGD update (same semantics as serial baseline).
*/

// ---------------- RNG (deterministic init only) ----------------
static unsigned long long g_seed = 42ULL;
static void set_seed_local(unsigned long long s){ g_seed = (s? s: 1ULL); }
static float rand_uniform(float a, float b){
    g_seed = 6364136223846793005ULL * g_seed + 1ULL;
    unsigned int r = (unsigned int)(g_seed >> 32);
    float u = (float)r / (float)0xFFFFFFFFu;
    return a + (b - a) * u;
}

// ---------------- Helpers ----------------
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
    float py = (y==0)? p0 : p1;
    if(py < eps) py = eps;
    return -logf(py);
}

// NCHW indexing helper
static inline int idx3(int c, int i, int j, int H, int W){
    return (c * H + i) * W + j;
}

// ---------------- Conv forward/back (3x3 pad=1) ----------------
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

// ---------------- Evaluation (serial, for correctness) ----------------
static float evaluate_accuracy(const Dataset *ds,
                               const float *Wc1, const float *bc1,
                               const float *Wc2, const float *bc2,
                               const float *Wfc, const float *bfc){
    int N = ds->N;
    int correct = 0;

    // Task-level parallelism: evaluation is independent per sample.
    // We must keep buffers thread-private to avoid overwrites.
#pragma omp parallel for reduction(+:correct) schedule(static)
    for(int n=0; n<N; n++){
        // Thread-private buffers (must be inside the loop or private)
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

        if(pred == (int)ds->y[n]) correct++;
    }

    return (float)correct / (float)N;
}


// ---------------- CLI ----------------
static void usage(const char *p){
    printf("Usage: %s -train data/train.bin -test data/test.bin [options]\n", p);
    printf("Options:\n");
    printf("  -epochs E   (default 5)\n");
    printf("  -batch B    (default 32)\n");
    printf("  -lr LR      (default 0.05)\n");
    printf("  -seed S     (default 42)\n");
    printf("  -log FILE   (default results/cnn_omp_log.csv)\n");
}

int main(int argc, char **argv){
    const char *train_path=NULL, *test_path=NULL;
    int epochs=5, batch=32;
    float lr=0.05f;
    unsigned long long seed=42ULL;
    const char *log_path="results/cnn_omp_log.csv";

    for(int i=1;i<argc;i++){
        if(!strcmp(argv[i],"-train") && i+1<argc) train_path=argv[++i];
        else if(!strcmp(argv[i],"-test") && i+1<argc) test_path=argv[++i];
        else if(!strcmp(argv[i],"-epochs") && i+1<argc) epochs=atoi(argv[++i]);
        else if(!strcmp(argv[i],"-batch") && i+1<argc) batch=atoi(argv[++i]);
        else if(!strcmp(argv[i],"-lr") && i+1<argc) lr=(float)atof(argv[++i]);
        else if(!strcmp(argv[i],"-seed") && i+1<argc) seed=(unsigned long long)atoll(argv[++i]);
        else if(!strcmp(argv[i],"-log") && i+1<argc) log_path=argv[++i];
        else { usage(argv[0]); return 1; }
    }
    if(!train_path || !test_path){ usage(argv[0]); return 1; }

    Dataset train,test;
    if(load_bin_dataset(train_path,&train)!=0){ fprintf(stderr,"Failed to load train\n"); return 1; }
    if(load_bin_dataset(test_path,&test)!=0){ fprintf(stderr,"Failed to load test\n"); return 1; }
    if(train.D!=1024 || test.D!=1024){ fprintf(stderr,"Expected D=1024\n"); return 1; }

    set_seed_local(seed);

    // Parameters
    float Wc1[8*1*3*3], bc1[8];
    float Wc2[16*8*3*3], bc2[16];
    float Wfc[2*1024], bfc[2];

    for(int i=0;i<8*1*3*3;i++) Wc1[i]=rand_uniform(-0.05f,0.05f);
    for(int i=0;i<16*8*3*3;i++) Wc2[i]=rand_uniform(-0.05f,0.05f);
    for(int i=0;i<2*1024;i++) Wfc[i]=rand_uniform(-0.05f,0.05f);
    memset(bc1,0,sizeof(bc1));
    memset(bc2,0,sizeof(bc2));
    memset(bfc,0,sizeof(bfc));

    // Global grads (after reduction)
    float dWc1[8*1*3*3], dbc1[8];
    float dWc2[16*8*3*3], dbc2[16];
    float dWfc[2*1024], dbfc[2];

    (void)system("mkdir -p results");
    FILE *logf=fopen(log_path,"w");
    if(!logf){ perror("fopen log"); return 1; }
    fprintf(logf,"epoch,epoch_time_s,train_loss,train_acc,test_acc,threads\n");

    int N = train.N;
    int max_threads = omp_get_max_threads();

    printf("OpenMP CNN: N=%d D=%d epochs=%d batch=%d lr=%.4f (max_threads=%d)\n",
           N, train.D, epochs, batch, lr, max_threads);

    double t_total0 = now_seconds();

    // Allocate per-thread gradient buffers (contiguous blocks)
    // Layout:
    //  dWc1_t [T][72], dbc1_t [T][8]
    //  dWc2_t [T][1152], dbc2_t [T][16]
    //  dWfc_t [T][2048], dbfc_t [T][2]
    float *dWc1_t = (float*)malloc((size_t)max_threads * (8*1*3*3) * sizeof(float));
    float *dbc1_t = (float*)malloc((size_t)max_threads * 8 * sizeof(float));
    float *dWc2_t = (float*)malloc((size_t)max_threads * (16*8*3*3) * sizeof(float));
    float *dbc2_t = (float*)malloc((size_t)max_threads * 16 * sizeof(float));
    float *dWfc_t = (float*)malloc((size_t)max_threads * (2*1024) * sizeof(float));
    float *dbfc_t = (float*)malloc((size_t)max_threads * 2 * sizeof(float));

    if(!dWc1_t || !dbc1_t || !dWc2_t || !dbc2_t || !dWfc_t || !dbfc_t){
        fprintf(stderr,"OOM thread grad buffers\n");
        return 1;
    }

    for(int e=1;e<=epochs;e++){
        double t0 = now_seconds();

        double loss_sum = 0.0;
        int correct = 0;

        for(int start=0; start<N; start+=batch){
            int bs = batch;
            if(start+bs > N) bs = N-start;

            // zero per-thread grads + stats for this batch
            memset(dWc1_t, 0, (size_t)max_threads*(8*1*3*3)*sizeof(float));
            memset(dbc1_t, 0, (size_t)max_threads*8*sizeof(float));
            memset(dWc2_t, 0, (size_t)max_threads*(16*8*3*3)*sizeof(float));
            memset(dbc2_t, 0, (size_t)max_threads*16*sizeof(float));
            memset(dWfc_t, 0, (size_t)max_threads*(2*1024)*sizeof(float));
            memset(dbfc_t, 0, (size_t)max_threads*2*sizeof(float));

            double batch_loss = 0.0;
            int batch_correct = 0;

            // Parallel over samples in the batch
#pragma omp parallel
            {
                int tid = omp_get_thread_num();

                float *dWc1p = &dWc1_t[(size_t)tid * (8*1*3*3)];
                float *dbc1p = &dbc1_t[(size_t)tid * 8];
                float *dWc2p = &dWc2_t[(size_t)tid * (16*8*3*3)];
                float *dbc2p = &dbc2_t[(size_t)tid * 16];
                float *dWfcp = &dWfc_t[(size_t)tid * (2*1024)];
                float *dbfcp = &dbfc_t[(size_t)tid * 2];

                // Thread-private activation/backprop buffers
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
                for(int ii=0; ii<bs; ii++){
                    int n = start + ii;
                    const float *x = &train.X[(size_t)n*1024];
                    int y = (int)train.y[n];

                    // Forward
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

                    // Backward: softmax + CE
                    float dz[2];
                    dz[0] = p0 - (y==0 ? 1.0f : 0.0f);
                    dz[1] = p1 - (y==1 ? 1.0f : 0.0f);

                    // FC backward
                    fc_backward(Wfc, pool2, dz, dWfcp, dbfcp, dtmp_flat);
                    memcpy(dpool2, dtmp_flat, sizeof(dpool2));

                    // Pool2 backward -> act2 grad
                    maxpool2x2_backward(dpool2, p2idx, 16, 16, 16, dact2);

                    // ReLU2 backward -> conv2 grad
                    relu_backward(conv2, dact2, dconv2, 16*16*16);

                    // Conv2 backward -> pool1 grad
                    conv3x3_pad1_backward(pool1, 8, 16, 16, Wc2, 16, dconv2, dWc2p, dbc2p, dx_pool1);
                    memcpy(dpool1, dx_pool1, sizeof(dpool1));

                    // Pool1 backward -> act1 grad
                    maxpool2x2_backward(dpool1, p1idx, 8, 32, 32, dact1);

                    // ReLU1 backward -> conv1 grad
                    relu_backward(conv1, dact1, dconv1, 8*32*32);

                    // Conv1 backward -> image grad (not used)
                    conv3x3_pad1_backward(x, 1, 32, 32, Wc1, 8, dconv1, dWc1p, dbc1p, dx_img);
                }

#pragma omp atomic
                batch_loss += local_loss;
#pragma omp atomic
                batch_correct += local_correct;
            } // end parallel region

            // Reduce thread grads into global grads
            memset(dWc1,0,sizeof(dWc1)); memset(dbc1,0,sizeof(dbc1));
            memset(dWc2,0,sizeof(dWc2)); memset(dbc2,0,sizeof(dbc2));
            memset(dWfc,0,sizeof(dWfc)); memset(dbfc,0,sizeof(dbfc));

            for(int t=0; t<max_threads; t++){
                float *p;

                p = &dWc1_t[(size_t)t*(8*1*3*3)];
                for(int i=0;i<(int)(8*1*3*3);i++) dWc1[i] += p[i];
                p = &dbc1_t[(size_t)t*8];
                for(int i=0;i<8;i++) dbc1[i] += p[i];

                p = &dWc2_t[(size_t)t*(16*8*3*3)];
                for(int i=0;i<(int)(16*8*3*3);i++) dWc2[i] += p[i];
                p = &dbc2_t[(size_t)t*16];
                for(int i=0;i<16;i++) dbc2[i] += p[i];

                p = &dWfc_t[(size_t)t*(2*1024)];
                for(int i=0;i<2*1024;i++) dWfc[i] += p[i];
                p = &dbfc_t[(size_t)t*2];
                dbfc[0] += p[0];
                dbfc[1] += p[1];
            }

            // SGD update (average grads)
            float inv_bs = 1.0f/(float)bs;

            for(int i=0;i<(int)(8*1*3*3);i++) Wc1[i] -= lr * dWc1[i]*inv_bs;
            for(int i=0;i<8;i++) bc1[i] -= lr * dbc1[i]*inv_bs;

            for(int i=0;i<(int)(16*8*3*3);i++) Wc2[i] -= lr * dWc2[i]*inv_bs;
            for(int i=0;i<16;i++) bc2[i] -= lr * dbc2[i]*inv_bs;

            for(int i=0;i<2*1024;i++) Wfc[i] -= lr * dWfc[i]*inv_bs;
            bfc[0] -= lr * dbfc[0]*inv_bs;
            bfc[1] -= lr * dbfc[1]*inv_bs;

            loss_sum += batch_loss;
            correct  += batch_correct;
        }

        double t1 = now_seconds();
        double epoch_time = t1 - t0;

        float train_loss = (float)(loss_sum / (double)N);
        float train_acc  = (float)correct / (float)N;
        float test_acc   = evaluate_accuracy(&test, Wc1, bc1, Wc2, bc2, Wfc, bfc);

        int threads_used = omp_get_max_threads();
        printf("Epoch %d: time=%.3fs loss=%.4f train_acc=%.4f test_acc=%.4f (threads=%d)\n",
               e, epoch_time, train_loss, train_acc, test_acc, threads_used);

        fprintf(logf, "%d,%.6f,%.6f,%.6f,%.6f,%d\n",
                e, epoch_time, train_loss, train_acc, test_acc, threads_used);
        fflush(logf);
    }

    double t_total1 = now_seconds();
    printf("Total training time: %.3fs\n", t_total1 - t_total0);

    fclose(logf);
    free(dWc1_t); free(dbc1_t); free(dWc2_t); free(dbc2_t); free(dWfc_t); free(dbfc_t);
    free_dataset(&train);
    free_dataset(&test);
    return 0;
}
