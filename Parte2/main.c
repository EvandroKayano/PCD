%%gpu

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define THREADS 16
#define N 2000
#define T 500
#define D 0.1f
#define DELTA_T 0.01f
#define DELTA_X 1.0f

__global__ void update_matriz(float* C, float* C_new, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // ignorando as bordas da matriz
    if (x > 0 && x < width - 1 && y > 0 && y < height - 1) {
        C_new[y * width + x] = C[y * width + x] + D * DELTA_T * (
            (C[y * width + (x + 1)] + C[y * width + (x - 1)] +
             C[(y + 1) * width + x] + C[(y - 1) * width + x] -
             4 * C[y * width + x]) / (DELTA_X * DELTA_X)
        );
    }
}

__global__ void update_dif_medio(float* C, float* C_new, float* difmedio, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // ignorando as bordas da matriz
    if (x > 0 && x < width - 1 && y > 0 && y < height - 1) {
        atomicAdd(difmedio, fabsf(C_new[y * width + x] - C[y * width + x]));
        C[y * width + x] = C_new[y * width + x];
    }
}

void diff(float* C, float* C_new, int width, int height) {
    float* d_C, * d_C_new, * d_difmedio;
    size_t size = width * height * sizeof(float);

    // Alocação da memória
    cudaMalloc(&d_C, size);
    cudaMalloc(&d_C_new, size);
    cudaMalloc(&d_difmedio, sizeof(float));

    cudaMemcpy(d_C, C, size, cudaMemcpyHostToDevice);

    // Define a dimensão dos blocos e da grade
    dim3 THREADS_PER_BLOCK(THREADS, THREADS);
    dim3 BLOCKS_PER_GRID((width + THREADS_PER_BLOCK.x - 1) / THREADS_PER_BLOCK.x, (height + THREADS_PER_BLOCK.y - 1) / THREADS_PER_BLOCK.y);

    printf("Utilizando %d threads.\n\n",THREADS);

    // Loop principal da simulação
    for (int t = 0; t < T; t++) {
        float difmedio = 0.0f;
        cudaMemcpy(d_difmedio, &difmedio, sizeof(float), cudaMemcpyHostToDevice);

        // Cálculo da Difusão
        update_matriz<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>(d_C, d_C_new, width, height);

        // Cálculo do dif_medio da iteração e atualização da matriz
        update_dif_medio<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>(d_C, d_C_new, d_difmedio, width, height);

        cudaDeviceSynchronize();

        if ((t % 100) == 0) {
            cudaMemcpy(&difmedio, d_difmedio, sizeof(float), cudaMemcpyDeviceToHost);
            printf("interacao %d - diferenca = %g\n", t, difmedio / ((N - 2) * (N - 2)));
        }
    }

    // Copia os resultados de volta para a CPU
    cudaMemcpy(C_new, d_C_new, size, cudaMemcpyDeviceToHost);

    // Libera a memória na GPU
    cudaFree(d_C);
    cudaFree(d_C_new);
    cudaFree(d_difmedio);
}


int main() {
    struct timespec start, end;
    float* C = (float*)calloc(N * N, sizeof(float));
    float* C_new = (float*)calloc(N * N, sizeof(float));
    C[(N / 2) * N + (N / 2)] = 1.0f;

    clock_gettime(CLOCK_MONOTONIC, &start);
    diff(C, C_new, N, N);
    clock_gettime(CLOCK_MONOTONIC, &end);

    printf("Concentração final no centro: %f\n", C_new[(N / 2) * N + (N / 2)]);

    double exec_time = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    printf("\nTempo de execução: %f segundos\n", exec_time);

    free(C);
    free(C_new);

    return 0;
}
