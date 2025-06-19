nvcc matrixMul.cu -O2 \
    -gencode arch=compute_61,code=sm_61 \
    -gencode arch=compute_61,code=compute_61 \
    -o matrixMul