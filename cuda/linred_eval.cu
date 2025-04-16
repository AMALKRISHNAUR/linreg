#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

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

// Read X_train
void readX(const char *filename, float *X, int rows, int cols, int features) {
    FILE *file = fopen(filename, "r");
    if (!file) {
        fprintf(stderr, "Error opening %s\n", filename);
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
        X[i * cols + cols - 1] = 1.0f; // Bias
    }
    fclose(file);
}

// Read y_train
void readY(const char *filename, float *y, int rows) {
    FILE *file = fopen(filename, "r");
    if (!file) {
        fprintf(stderr, "Error opening %s\n", filename);
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

// Read weights
void readWeights(const char *filename, float *beta, int size) {
    FILE *file = fopen(filename, "r");
    if (!file) {
        fprintf(stderr, "Error opening %s\n", filename);
        exit(1);
    }
    char line[1024];
    fgets(line, 1024, file);
    for (int i = 0; i < size; i++) {
        char label[10];
        float value;
        if (fscanf(file, "%[^,],%f\n", label, &value) != 2) {
            fprintf(stderr, "Error reading weight %d in %s\n", i, filename);
            fclose(file);
            exit(1);
        }
        beta[i] = value;
    }
    fclose(file);
}

// CUDA kernel for prediction (y_pred = X * beta)
__global__ void predictKernel(float *X, float *beta, float *y_pred, int rows, int cols) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < rows) {
        float sum = 0.0f;
        for (int j = 0; j < cols; j++) {
            sum += X[i * cols + j] * beta[j];
        }
        y_pred[i] = sum;
    }
}

// CUDA kernel for computing squared differences
__global__ void squaredDiffKernel(float *y_true, float *y_pred, float *sq_diff, int rows) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < rows) {
        float diff = y_true[i] - y_pred[i];
        sq_diff[i] = diff * diff;
    }
}

// CUDA kernel for reduction (sum)
__global__ void reductionKernel(float *input, float *output, int n) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Load data
    sdata[tid] = (i < n) ? input[i] : 0.0f;
    __syncthreads();

    // Reduce
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // Write result
    if (tid == 0) {
        output[blockIdx.x] = sdata[0];
    }
}

// Compute metrics
void computeMetrics(float *d_y_true, float *d_y_pred, float *h_y_true, float *h_y_pred, int rows, float *rmse, float *r2) {
    float *d_sq_diff, *d_sum;
    CUDA_CHECK(cudaMalloc(&d_sq_diff, rows * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_sum, ((rows + 255) / 256) * sizeof(float)));

    // Compute squared differences
    int threads = 256;
    int blocks = (rows + threads - 1) / threads;
    squaredDiffKernel<<<blocks, threads>>>(d_y_true, d_y_pred, d_sq_diff, rows);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Sum squared differences
    reductionKernel<<<blocks, threads, threads * sizeof(float)>>>(d_sq_diff, d_sum, rows);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy partial sums to host
    float *h_sum = (float *)malloc(blocks * sizeof(float));
    CUDA_CHECK(cudaMemcpy(h_sum, d_sum, blocks * sizeof(float), cudaMemcpyDeviceToHost));

    // Final sum
    float ss_res = 0.0f;
    for (int i = 0; i < blocks; i++) {
        ss_res += h_sum[i];
    }
    *rmse = sqrt(ss_res / rows);

    // Compute y_mean
    float y_mean = 0.0f;
    for (int i = 0; i < rows; i++) {
        y_mean += h_y_true[i];
    }
    y_mean /= rows;

    // Compute ss_tot
    float *d_diff_mean, *d_sq_diff_mean;
    CUDA_CHECK(cudaMalloc(&d_diff_mean, rows * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_sq_diff_mean, rows * sizeof(float)));

    // Compute y - y_mean
    for (int i = 0; i < rows; i++) {
        h_y_pred[i] = y_mean; // Reuse h_y_pred
    }
    CUDA_CHECK(cudaMemcpy(d_y_pred, h_y_pred, rows * sizeof(float), cudaMemcpyHostToDevice));
    squaredDiffKernel<<<blocks, threads>>>(d_y_true, d_y_pred, d_sq_diff_mean, rows);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Sum
    reductionKernel<<<blocks, threads, threads * sizeof(float)>>>(d_sq_diff_mean, d_sum, rows);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_sum, d_sum, blocks * sizeof(float), cudaMemcpyDeviceToHost));
    float ss_tot = 0.0f;
    for (int i = 0; i < blocks; i++) {
        ss_tot += h_sum[i];
    }

    *r2 = 1.0f - (ss_res / ss_tot);

    // Cleanup
    free(h_sum);
    CUDA_CHECK(cudaFree(d_sq_diff));
    CUDA_CHECK(cudaFree(d_sum));
    CUDA_CHECK(cudaFree(d_diff_mean));
    CUDA_CHECK(cudaFree(d_sq_diff_mean));
}

// Print predictions
void printPredictions(float *y_pred, float *y_true, int rows, int max_rows) {
    printf("Sample Predictions vs Actual (first %d):\n", max_rows);
    printf("i\tPredicted\tActual\n");
    for (int i = 0; i < max_rows && i < rows; i++) {
        printf("%d\t%.4f\t\t%.4f\n", i, y_pred[i], y_true[i]);
    }
}

int main() {
    const int rows = 256;
    const int features = 13;
    const int cols = features + 1;

    // Host memory
    float *h_X = (float *)malloc(rows * cols * sizeof(float));
    float *h_y = (float *)malloc(rows * sizeof(float));
    float *h_beta = (float *)malloc(cols * sizeof(float));
    float *h_y_pred = (float *)malloc(rows * sizeof(float));

    if (!h_X || !h_y || !h_beta || !h_y_pred) {
        fprintf(stderr, "Memory allocation failed\n");
        exit(1);
    }

    // Initialize X
    for (int i = 0; i < rows * cols; i++) {
        h_X[i] = 0.0f;
    }

    // Read data
    printf("Reading X_train.csv\n");
    readX("X_train.csv", h_X, rows, cols, features);
    printf("Reading y_train.csv\n");
    readY("y_train.csv", h_y, rows);
    printf("Reading weights.csv\n");
    readWeights("weights.csv", h_beta, cols);

    // Device memory
    float *d_X, *d_y, *d_beta, *d_y_pred;
    CUDA_CHECK(cudaMalloc(&d_X, rows * cols * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_y, rows * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_beta, cols * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_y_pred, rows * sizeof(float)));

    // Copy to device
    CUDA_CHECK(cudaMemcpy(d_X, h_X, rows * cols * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_y, h_y, rows * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_beta, h_beta, cols * sizeof(float), cudaMemcpyHostToDevice));

    // Predict
    int threads = 256;
    int blocks = (rows + threads - 1) / threads;
    predictKernel<<<blocks, threads>>>(d_X, d_beta, d_y_pred, rows, cols);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Compute metrics
    float rmse, r2;
    computeMetrics(d_y, d_y_pred, h_y, h_y_pred, rows, &rmse, &r2);

    // Copy predictions to host
    CUDA_CHECK(cudaMemcpy(h_y_pred, d_y_pred, rows * sizeof(float), cudaMemcpyDeviceToHost));

    // Print results
    printPredictions(h_y_pred, h_y, rows, 5);
    printf("Training RMSE: %.4f\n", rmse);
    printf("Training R²: %.4f\n", r2);

    // Cleanup
    free(h_X); free(h_y); free(h_beta); free(h_y_pred);
    CUDA_CHECK(cudaFree(d_X)); CUDA_CHECK(cudaFree(d_y));
    CUDA_CHECK(cudaFree(d_beta)); CUDA_CHECK(cudaFree(d_y_pred));

    return 0;
}