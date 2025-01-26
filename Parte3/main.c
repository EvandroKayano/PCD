#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>

#define N 2000  // Tamanho da grade
#define T 500   // Número de iterações no tempo
#define D 0.1   // Coeficiente de difusão
#define DELTA_T 0.01
#define DELTA_X 1.0

void diff_eq(double **C, double **C_new, int first_row, int last_row, int rank, int size, MPI_Comm comm) {
    MPI_Request request[4];
    int num_requests = 0;

    for (int t = 0; t < T; t++) { // iterações
        num_requests = 0;

        // primeira e ultima linhas
        if (rank > 0) { // Manda a primeira linha local pro processo anterior
            MPI_Isend(C[1], N, MPI_DOUBLE, rank - 1, 0, comm, &request[num_requests++]);
            MPI_Irecv(C[0], N, MPI_DOUBLE, rank - 1, 0, comm, &request[num_requests++]);
        }
        if (rank < size - 1) { // Manda a última linha local pro processo seguinte
            MPI_Isend(C[last_row - first_row], N, MPI_DOUBLE, rank + 1, 0, comm, &request[num_requests++]);
            MPI_Irecv(C[last_row - first_row + 1], N, MPI_DOUBLE, rank + 1, 0, comm, &request[num_requests++]);
        }

        // Sincroniza as trocas de mensagens
        if (num_requests > 0) {
            MPI_Waitall(num_requests, request, MPI_STATUS_IGNORE);
        }

        // Cálculo da difusão
        for (int i = 1; i <= last_row - first_row; i++) {
            for (int j = 1; j < N - 1; j++) {
                C_new[i][j] = C[i][j] + D * DELTA_T * (
                    (C[i+1][j] + C[i-1][j] + C[i][j+1] + C[i][j-1] - 4 * C[i][j]) / (DELTA_X * DELTA_X)
                );
            }
        }

        // Atualizar a matriz para a próxima iteração e cálculo do difmédio
        double difmedio = 0.;
        for (int i = 1; i <= last_row - first_row; i++) {
            for (int j = 1; j < N - 1; j++) {
                difmedio += fabs(C_new[i][j] - C[i][j]);
                C[i][j] = C_new[i][j];
            }
        }

        if ((t % 100) == 0 && rank == 0) {
            printf("interação %d - diferenca = %g\n", t, difmedio / ((N - 2) * (N - 2)));
        }
    }
}

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int linhas_proc = N / size;  // n de linhas por processo
    int first_row = rank * linhas_proc;
    int last_row = (rank + 1) * linhas_proc - 1;

    // Verifica se o tamanho é divisível por size e ajustar o último processo
    if (rank == size - 1) {
        last_row = N - 1;
        linhas_proc = last_row - first_row + 1;
    }

    // Alocação de memória
    double **C = (double **)malloc((linhas_proc + 2) * sizeof(double *));  // +2 para fronteiras -> 1 antes e 1 depois
    double **C_new = (double **)malloc((linhas_proc + 2) * sizeof(double *));
    for (int i = 0; i < linhas_proc + 2; i++) {
        C[i] = calloc(N,sizeof(double));
        C_new[i] = calloc(N, sizeof(double));
    }

    // Proc pai configura
    if (rank == 0) {
        C[linhas_proc / 2][N / 2] = 1.0;
    }

    // Proc pai verifica o tempo de exec
    double start_time, end_time;
    if (rank == 0) {
        start_time = MPI_Wtime();
    }

    diff_eq(C, C_new, first_row, last_row, rank, size, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Concentração final no centro: %f\n", C[linhas_proc / 2][N / 2]);
    }

    if (rank == 0) {
        end_time = MPI_Wtime();
        printf("Tempo de execução: %f segundos\n", end_time - start_time);
    }

    // Liberar memória
    for (int i = 0; i < linhas_proc + 2; i++) {
        free(C[i]);
        free(C_new[i]);
    }
    free(C);
    free(C_new);

    MPI_Finalize();
    return 0;
}
