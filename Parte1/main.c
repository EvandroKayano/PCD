# include <stdio.h>
# include <stdlib.h>
# include <omp.h>
# include <math.h>
# define N 2000  // Tamanho da grade
# define T 500 // Número de iterações
# define D 0.1  // Coeficiente de difusão
# define DELTA_T 0.01
# define DELTA_X 1.0

void diff_eq(double** C, double** C_new) {
    for (int t = 0; t < T; t++) { // iterações
        for (int i = 1; i < N - 1; i++) { // linhas
            for (int j = 1; j < N - 1; j++) { // colunas

                C_new[i][j] = C[i][j] + D * DELTA_T * (
                    (C[i+1][j] + C[i-1][j] + C[i][j+1] + C[i][j-1] - 4 * C[i][j]) / (DELTA_X * DELTA_X)
                );

            }
        }
        // Atualizar matriz para a próxima iteração
        double difmedio = 0.;
        for (int i = 1; i < N - 1; i++) {
            for (int j = 1; j < N - 1; j++) {
                difmedio += fabs(C_new[i][j] - C[i][j]);
                C[i][j] = C_new[i][j];
            }
        }
        if ((t%100) == 0)
          printf("interação %d - diferença = %g\n", t, difmedio/((N-2)*(N-2)));
    }
}

void diff_openmp(double** C, double** C_new, int threads) {
    omp_set_num_threads(threads);
    for (int t = 0; t < T; t++) { // iterações

    #pragma omp parallel for collapse(2) // collapse vai pegar os 2 próximos for's

        for (int i = 1; i < N - 1; i++) { // linhas
            for (int j = 1; j < N - 1; j++) { // colunas
                C_new[i][j] = C[i][j] + D * DELTA_T * (
                    (C[i+1][j] + C[i-1][j] + C[i][j+1] + C[i][j-1] - 4 * C[i][j]) / (DELTA_X * DELTA_X)
                );
            }
        }
        double difmedio = 0.;
        #pragma omp parallel for collapse(2) reduction(+:difmedio)
        // Atualizar matriz para a próxima iteração
        for (int i = 1; i < N - 1; i++) {
            for (int j = 1; j < N - 1; j++) {
                difmedio += fabs(C_new[i][j] - C[i][j]);
                C[i][j] = C_new[i][j];
            }
        }
        if ((t%100) == 0)
          printf("interação %d - diferença = %g\n", t, difmedio/((N-2)*(N-2)));
    }

}

int main() {
    int threads = 32;
    double **C_0 = malloc(N * sizeof(double*));      // Concentração inicial
    double **C_ser = malloc(N * sizeof(double*));  // Concentração para a próxima iteração
    for (size_t i = 0; i < N; i++)
    {
        C_0[i] = calloc(N , sizeof(double));
        C_ser[i] = calloc(N , sizeof(double));
    }
    
    double **C_1 = malloc(N * sizeof(double*));      
    double **C_openmp = malloc(N * sizeof(double*));  
    for (size_t i = 0; i < N; i++)
    {
        C_1[i] = calloc(N , sizeof(double));
        C_openmp[i] = calloc(N , sizeof(double));
    }

    double start_serial, end_serial, start_openmp, end_openmp;
    C_0[N/2][N/2] = 1.0;
    C_1[N/2][N/2] = 1.0;


    // SERIALIZADO
    start_serial = omp_get_wtime();
    diff_eq(C_0, C_ser);
    end_serial = omp_get_wtime();
    printf("\nConcentração final no centro da equação serializada: %.6f\n", C_0[N/2][N/2]);
    printf("Tempo de execução: %.6f\n\n",end_serial-start_serial);

    
    // PARALELIZADO OPENMP
    start_openmp = omp_get_wtime();
    diff_openmp(C_1,C_openmp,threads);
    end_openmp = omp_get_wtime();

    printf("\nConcentração final no centro da equação paralelizada em OpenMP: %.6f\n", C_1[N/2][N/2]);
    printf("Tempo de execução: %.6f\n\n",end_openmp-start_openmp);

    for (size_t i = 0; i < N; i++)
    {
        free(C_0[i]);
        free(C_ser[i]);

        free(C_1[i]);
        free(C_openmp[i]);
    }
    free(C_0);
    free(C_ser);
    free(C_1);
    free(C_openmp);

    return 0;
}
