# Parallelization of Deep Learning Model Training (CNN)

## Dataset
This project uses the TFDS Malaria dataset, preprocessed to 32x32 grayscale and exported as binary `.bin` files:
- `data/malaria_train_32.bin`
- `data/malaria_test_32.bin`

(These `.bin` files are not tracked in git due to size. Generate them using the provided Colab notebook.)

## Build
```bash
make clean
make
Run: Serial baseline
bash
Copy code
./bin/cnn_serial -train data/malaria_train_32.bin -test data/malaria_test_32.bin -epochs 5 -batch 32 -lr 0.05
Run: Chosen parallel strategy (OpenMP)
OpenMP data-parallel training + parallel evaluation (recommended setting on this machine: 4 threads)

bash
Copy code
export OMP_PROC_BIND=true
export OMP_PLACES=cores
export OMP_NUM_THREADS=4

./bin/cnn_omp -train data/malaria_train_32.bin -test data/malaria_test_32.bin -epochs 5 -batch 32 -lr 0.05 -log results/final_omp_evalpar_t4.csv
