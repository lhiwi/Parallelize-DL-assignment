```markdown
# Parallelization of Deep Learning Model Training (CNN) — Serial vs OpenMP

This repository contains a from-scratch CNN training pipeline in **C**, plus an **OpenMP** parallel version that accelerates training on a shared-memory CPU. The task is **binary classification** on the **Malaria** cell image dataset (TFDS), exported into compact `.bin` files for fast, reproducible experiments.

---

## Repository Structure

```

.
├── bin/                  # compiled executables (created by build)
├── data/                 # dataset binaries (.bin)
├── results/              # CSV logs (timings, loss, accuracy)
├── src/                  # C sources
├── Makefile
└── README.md

````

---

## Prerequisites

- Ubuntu 22.04 (or similar Linux)
- GCC with OpenMP support
- Make

Check tools:

```bash
gcc --version
make --version
````

OpenMP is included with GCC on Ubuntu.

---

## Dataset (Required)

Experiments use two exported binary files:

* `data/malaria_train_32.bin`
* `data/malaria_test_32.bin`

### Binary Format (for reproducibility)

Each `.bin` file stores:

1. `int32 N`  (number of samples)
2. `int32 D`  (feature dimension; expected `D = 1024`)
3. `float32 X[N*D]` (row-major features)
4. `uint8 y[N]`     (labels 0/1)

### Preprocessing Summary

Images are:

* converted to grayscale
* resized to `32×32`
* normalized to `[0,1]`
* flattened to `D = 1024`

> If the `.bin` files are not present, generate them using the provided Colab/preprocessing notebook (or your documented export script). The C code expects the exact format above.

---

## Build

Clean and compile:

```bash
make clean
make
```

Executables will be created in `bin/`:

* `bin/cnn_serial`
* `bin/cnn_omp`

If your Makefile uses specific targets:

```bash
make cnn_serial
make cnn_omp
```

---

## Run Experiments

### 1) Serial Baseline (CNN)

Runs 5 epochs and writes the default log to `results/cnn_serial_log.csv`:

```bash
./bin/cnn_serial \
  -train data/malaria_train_32.bin \
  -test  data/malaria_test_32.bin  \
  -epochs 5 -batch 32 -lr 0.05
```

Expected output includes per-epoch:

* epoch time
* train loss
* train accuracy
* test accuracy

---

### 2) Parallel Version (Chosen Strategy: OpenMP)

The parallel implementation uses **data parallelism** within each mini-batch:

* each thread processes different samples
* gradients are accumulated thread-locally
* gradients are reduced into one batch gradient
* **one** SGD update per mini-batch (same semantics as serial)

Pinned placement is recommended for stable timing:

```bash
export OMP_PROC_BIND=true
export OMP_PLACES=cores
export OMP_NUM_THREADS=4
```

Run and save a log CSV:

```bash
./bin/cnn_omp \
  -train data/malaria_train_32.bin \
  -test  data/malaria_test_32.bin  \
  -epochs 5 -batch 32 -lr 0.05 \
  -log results/cnn_omp_evalpar_pin_t4.csv
```

---

## Reproducing Scalability Measurements

To reproduce the thread-scaling study (example: 1, 4, 8, 12, 14 threads):

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

## Output Logs

All logs are stored as CSV files under `results/`.
CSV header format:

```
epoch,epoch_time_s,train_loss,train_acc,test_acc
```

Example:

```bash
head -n 10 results/cnn_serial_log.csv
head -n 10 results/cnn_omp_evalpar_pin_t4.csv
```

---

## Notes on Correctness

Correctness is verified by checking that:

* training loss decreases across epochs
* accuracy improves across epochs
* serial and OpenMP runs show comparable learning behavior

Small differences can occur due to floating-point reduction order in parallel accumulation.

---

## Common Issues

### OpenMP threads not changing

Verify runtime thread count by checking program output and ensure:

```bash
echo $OMP_NUM_THREADS
```

### Missing dataset files

Ensure `data/malaria_train_32.bin` and `data/malaria_test_32.bin` exist:

```bash
ls -lh data
```

---

## License / Academic Use

This code is provided for educational purposes as part of a distributed computing take-home assignment.

```
```
