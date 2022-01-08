echo
echo "Building parallel_omp_test"
gcc -Wall -fopenmp tests/parallel_omp_test.c -o bin/parallel_omp_test
echo "Done. Executable will be in bin/parallel_omp_test"
echo
echo "Building parallel_cuda_test"
nvcc tests/parallel_cuda_test.cu -o bin/parallel_cuda_test
echo "Done. Executable will be in bin/parallel_cuda_test"
echo