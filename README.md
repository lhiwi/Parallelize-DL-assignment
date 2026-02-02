
# Parallelization of Deep Learning Model Training (CNN) — Serial vs OpenMP

This repository contains a from-scratch **CNN training pipeline in C** and an **OpenMP parallel** implementation for accelerating training on a shared-memory CPU. The task is **binary classification** on the Malaria cell image dataset, exported into compact binary files to ensure fast loading and reproducible timing.

---

## 1) Requirements

**Software**
- Ubuntu 22.04 (or similar Linux)
- GCC (with OpenMP support)
- GNU Make

Check:
```bash
gcc --version
make --version
````

---

## 2) Dataset (Binary Files)

Experiments expect two binary files in `data/`:

* `data/malaria_train_32.bin`
* `data/malaria_test_32.bin`

### Binary format (for reproducibility)

Each `.bin` file stores:

1. `int32 N`  (number of samples)
2. `int32 D`  (feature dimension; expected `D = 1024`)
3. `float32 X[N*D]` (row-major feature matrix)
4. `uint8 y[N]`     (labels in {0,1})

### Preprocessing summary

Images are:

* converted to grayscale
* resized to `32×32`
* normalized to `[0,1]`
* flattened to `D = 1024`

> If the `.bin` files are not present, generate them using the documented preprocessing script/notebook that exports the exact format above.

---

## 3) Build (Compile)

Clean and compile:

```bash
make clean
make
```

Expected executables:

* `bin/cnn_serial`
* `bin/cnn_omp`

---

## 4) Run: Serial Baseline (CNN)

Runs 5 epochs and prints per-epoch time, loss, and accuracies.
A CSV log is written to:

* `results/cnn_serial_log.csv`

Command:

```bash
mkdir -p results

./bin/cnn_serial \
  -train data/malaria_train_32.bin \
  -test  data/malaria_test_32.bin  \
  -epochs 5 -batch 32 -lr 0.05
```

Inspect the log:

```bash
head -n 10 results/cnn_serial_log.csv
```

---

## 5) Run: Parallel Version (Chosen Strategy: OpenMP)

### Parallelization strategy (clarified)

* **Training (required parallel version): Data parallelism with OpenMP.**
  Mini-batch samples are distributed across threads. Each thread computes forward/backward passes and accumulates thread-local gradients. Gradients are reduced into one batch gradient, and **one SGD update per mini-batch** is applied (same semantics as serial).

* **Evaluation (extra optimization): Task parallelism with OpenMP.**
  Test accuracy computation is parallelized across samples using OpenMP reduction, improving end-to-end epoch time without changing training updates.

### Recommended environment (stable timing)

```bash
export OMP_PROC_BIND=true
export OMP_PLACES=cores
export OMP_NUM_THREADS=4
```

Run and save a CSV log:

```bash
mkdir -p results

./bin/cnn_omp \
  -train data/malaria_train_32.bin \
  -test  data/malaria_test_32.bin  \
  -epochs 5 -batch 32 -lr 0.05 \
  -log results/cnn_omp_evalpar_pin_t4.csv
```

Inspect the log:

```bash
head -n 10 results/cnn_omp_evalpar_pin_t4.csv
```

---

## 6) Reproduce Scalability Experiments (Thread Scaling)

This reproduces the scaling study for multiple thread counts:

```bash
mkdir -p results
export OMP_PROC_BIND=true
export OMP_PLACES=cores

for t in 1 4 8 12 14; do
  export OMP_NUM_THREADS=$t
  ./bin/cnn_omp \
    -train data/malaria_train_32.bin \
    -test  data/malaria_test_32.bin  \
    -epochs 5 -batch 32 -lr 0.05 \
    -log results/cnn_omp_evalpar_pin_t${t}.csv
done
```

---

## 7) Correctness Criteria

Correctness is verified by checking that:

* training loss decreases across epochs
* accuracy improves across epochs
* serial and OpenMP runs show comparable learning behavior

Small numerical differences can occur due to floating-point reduction order in parallel gradient accumulation.

---

```

