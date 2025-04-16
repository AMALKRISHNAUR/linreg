#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

// Error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error in %s:%d: %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

#define CUBLAS_CHECK(call) \
    do { \
        cublasStatus_t status = call; \
        if (status != CUBLAS_STATUS_SUCCESS) { \
            fprintf(stderr, "cuBLAS error in %s:%d: %d\n", \
                    __FILE__, __LINE__, status); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

#define TILE_SIZE 16

// Print matrix function
void printMatrix(const char *name, float *matrix, int rows, int cols, int max_rows) {
    printf("%s (%dx%d):\n", name, rows, cols);
    int print_rows = (max_rows > 0 && max_rows < rows) ? max_rows : rows;
    for (int i = 0; i < print_rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%.4f ", matrix[i * cols + j]);
        }
        printf("\n");
    }
    if (print_rows < rows) {
        printf("... (%d more rows)\n", rows - print_rows);
    }
    printf("\n");
}

// Save weights to file
void saveWeights(const char *filename, float *beta, int size) {
    FILE *file = fopen(filename, "w");
    if (!file) {
        fprintf(stderr, "Error opening file %s for writing\n", filename);
        exit(1);
    }
    fprintf(file, "weight,value\n");
    for (int i = 0; i < size - 1; i++) {
        fprintf(file, "w%d,%.6f\n", i + 1, beta[i]);
    }
    fprintf(file, "b,%.6f\n", beta[size - 1]);
    fclose(file);
    printf("Weights saved to %s\n", filename);
}

// Tiled matrix multiplication kernel
__global__ void tiledMatrixMulKernel(float *A, float *B, float *C, int m, int n, int p) {
    __shared__ float s_A[TILE_SIZE][TILE_SIZE];
    __shared__ float s_B[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float sum = 0.0f;

    for (int t = 0; t < (n + TILE_SIZE - 1) / TILE_SIZE; t++) {
        int a_col = t * TILE_SIZE + threadIdx.x;
        if (row < m && a_col < n) {
            s_A[threadIdx.y][threadIdx.x] = A[row * n + a_col];
        } else {
            s_A[threadIdx.y][threadIdx.x] = 0.0f;
        }

        int b_row = t * TILE_SIZE + threadIdx.y;
        if (col < p && b_row < n) {
            s_B[threadIdx.y][threadIdx.x] = B[b_row * p + col];
        } else {
            s_B[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        if (row < m && col < p) {
            for (int k = 0; k < TILE_SIZE && k < n; k++) {
                sum += s_A[threadIdx.y][k] * s_B[k][threadIdx.x];
            }
        }

        __syncthreads();
    }

    if (row < m && col < p) {
        C[row * p + col] = sum;
    }
}

// Kernel to compute X^T y
__global__ void computeXTyKernel(float *X, float *y, float *XTy, int rows, int cols) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < cols) {
        float sum = 0.0f;
        for (int i = 0; i < rows; i++) {
            sum += X[i * cols + col] * y[i];
        }
        XTy[col] = sum;
    }
}

// Read X
void readX(const char *filename, float *X, int rows, int cols, int features) {
    FILE *file = fopen(filename, "r");
    if (!file) {
        fprintf(stderr, "Error opening file %s\n", filename);
        exit(1);
    }
    char line[1024];
    fgets(line, 1024, file);
    for (int i = 0; i < rows; i++) {
        if (fscanf(file, "%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f",
                   &X[i * cols + 0], &X[i * cols + 1], &X[i * cols + 2], &X[i * cols + 3],
                   &X[i * cols + 4], &X[i * cols + 5], &X[i * cols + 6], &X[i * cols + 7],
                   &X[i * cols + 8], &X[i * cols + 9], &X[i * cols + 10], &X[i * cols + 11],
                   &X[i * cols + 12]) != features) {
            fprintf(stderr, "Error reading row %d in %s\n", i, filename);
            fclose(file);
            exit(1);
        }
    }
    fclose(file);
    for (int i = 0; i < rows; i++) {
        X[i * cols + cols - 1] = 1.0f;
    }
}

// Read y
void readY(const char *filename, float *y, int rows) {
    FILE *file = fopen(filename, "r");
    if (!file) {
        fprintf(stderr, "Error opening file %s\n", filename);
        exit(1);
    }
    char line[1024];
    fgets(line, 1024, file);
    for (int i = 0; i < rows; i++) {
        if (fscanf(file, "%f", &y[i]) != 1) {
            fprintf(stderr, "Error reading row %d in %s\n", i, filename);
            fclose(file);
            exit(1);
        }
    }
    fclose(file);
}

int main() {
    const int rows = 256;
    const int features = 13;
    const int cols = features + 1;
    const char *x_filename = "X_train.csv";
    const char *y_filename = "y_train.csv";
    const char *weights_filename = "weights.csv";

    // Host memory
    float *h_X = (float *)malloc(rows * cols * sizeof(float));
    float *h_y = (float *)malloc(rows * sizeof(float));
    float *h_XTX = (float *)malloc(cols * cols * sizeof(float));
    float *h_XTX_inv = (float *)malloc(cols * cols * sizeof(float));
    float *h_XTy = (float *)malloc(cols * sizeof(float));
    float *h_beta = (float *)calloc(cols, sizeof(float));

    if (!h_X || !h_y || !h_XTX || !h_XTX_inv || !h_XTy || !h_beta) {
        fprintf(stderr, "Host memory allocation failed\n");
        exit(1);
    }

    for (int i = 0; i < rows * cols; i++) {
        h_X[i] = 0.0f;
    }

    // Read datasets
    printf("Reading X from %s\n", x_filename);
    readX(x_filename, h_X, rows, cols, features);
    printf("Reading y from %s\n", y_filename);
    readY(y_filename, h_y, rows);

    // Print X and y
    printMatrix("X", h_X, rows, cols, 5);
    printMatrix("y", h_y, rows, 1, 5);

    // Device memory
    float *d_X, *d_y, *d_XTX, *d_XTX_inv, *d_XTy, *d_beta;
    float **d_XTX_array, **d_XTX_inv_array;
    CUDA_CHECK(cudaMalloc(&d_X, rows * cols * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_y, rows * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_XTX, cols * cols * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_XTX_inv, cols * cols * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_XTy, cols * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_beta, cols * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_XTX_array, sizeof(float *)));
    CUDA_CHECK(cudaMalloc(&d_XTX_inv_array, sizeof(float *)));

    // Set pointer arrays
    CUDA_CHECK(cudaMemcpy(d_XTX_array, &d_XTX, sizeof(float *), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_XTX_inv_array, &d_XTX_inv, sizeof(float *), cudaMemcpyHostToDevice));

    // Copy data
    printf("Copying X to device\n");
    CUDA_CHECK(cudaMemcpy(d_X, h_X, rows * cols * sizeof(float), cudaMemcpyHostToDevice));
    printf("Copying y to device\n");
    CUDA_CHECK(cudaMemcpy(d_y, h_y, rows * sizeof(float), cudaMemcpyHostToDevice));

    // Initialize cuBLAS
    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    // Compute X^T X
    printf("Computing X^T X\n");
    dim3 blockSize(TILE_SIZE, TILE_SIZE);
    dim3 gridSize((cols + TILE_SIZE - 1) / TILE_SIZE, (cols + TILE_SIZE - 1) / TILE_SIZE);
    tiledMatrixMulKernel<<<gridSize, blockSize>>>(d_X, d_X, d_XTX, cols, rows, cols);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_XTX, d_XTX, cols * cols * sizeof(float), cudaMemcpyDeviceToHost));
    printMatrix("X^T X", h_XTX, cols, cols, 0);

    // Compute X^T y
    printf("Computing X^T y\n");
    blockSize = dim3(256, 1);
    gridSize = dim3((cols + blockSize.x - 1) / blockSize.x, 1);
    computeXTyKernel<<<gridSize, blockSize>>>(d_X, d_y, d_XTy, rows, cols);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_XTy, d_XTy, cols * sizeof(float), cudaMemcpyDeviceToHost));
    printMatrix("X^T y", h_XTy, cols, 1, 0);

    // Matrix inversion
    printf("Inverting X^T X\n");
    int *d_pivotArray, *d_infoArray;
    CUDA_CHECK(cudaMalloc(&d_pivotArray, cols * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_infoArray, sizeof(int)));

    CUBLAS_CHECK(cublasSgetrfBatched(handle, cols, d_XTX_array, cols, d_pivotArray, d_infoArray, 1));
    CUDA_CHECK(cudaDeviceSynchronize());

    int h_info;
    CUDA_CHECK(cudaMemcpy(&h_info, d_infoArray, sizeof(int), cudaMemcpyDeviceToHost));
    if (h_info != 0) {
        fprintf(stderr, "LU decomposition failed: info=%d\n", h_info);
        exit(1);
    }

    CUBLAS_CHECK(cublasSgetriBatched(handle, cols, (const float **)d_XTX_array, cols, d_pivotArray,
                                     d_XTX_inv_array, cols, d_infoArray, 1));
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(&h_info, d_infoArray, sizeof(int), cudaMemcpyDeviceToHost));
    if (h_info != 0) {
        fprintf(stderr, "Matrix inversion failed: info=%d\n", h_info);
        exit(1);
    }

    CUDA_CHECK(cudaMemcpy(h_XTX_inv, d_XTX_inv, cols * cols * sizeof(float), cudaMemcpyDeviceToHost));
    printMatrix("(X^T X)^(-1)", h_XTX_inv, cols, cols, 0);

    // Compute beta
    printf("Computing beta\n");
    blockSize = dim3(TILE_SIZE, TILE_SIZE);
    gridSize = dim3(1, cols / TILE_SIZE + 1);
    tiledMatrixMulKernel<<<gridSize, blockSize>>>(d_XTX_inv, d_XTy, d_beta, cols, cols, 1);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_beta, d_beta, cols * sizeof(float), cudaMemcpyDeviceToHost));
    printMatrix("beta", h_beta, cols, 1, 0);

    // Print weights explicitly
    printf("Linear Regression Coefficients:\n");
    for (int i = 0; i < features; i++) {
        printf("Weight w%d (for feature %d) = %.4f\n", i + 1, i + 1, h_beta[i]);
    }
    printf("Bias b = %.4f\n", h_beta[cols - 1]);

    // Save weights
    saveWeights(weights_filename, h_beta, cols);

    // Cleanup
    free(h_X); free(h_y); free(h_XTX); free(h_XTX_inv); free(h_XTy); free(h_beta);
    CUDA_CHECK(cudaFree(d_X)); CUDA_CHECK(cudaFree(d_y));
    CUDA_CHECK(cudaFree(d_XTX)); CUDA_CHECK(cudaFree(d_XTX_inv));
    CUDA_CHECK(cudaFree(d_XTy)); CUDA_CHECK(cudaFree(d_beta));
    CUDA_CHECK(cudaFree(d_XTX_array)); CUDA_CHECK(cudaFree(d_XTX_inv_array));
    CUDA_CHECK(cudaFree(d_pivotArray)); CUDA_CHECK(cudaFree(d_infoArray));
    CUBLAS_CHECK(cublasDestroy(handle));

    return 0;
}