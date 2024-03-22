#include <stdio.h>
#include <gsl/gsl_linalg.h>

void print_matrix(gsl_matrix *m) {
    for (size_t i = 0; i < m->size1; ++i) {
        for (size_t j = 0; j < m->size2; ++j) {
            printf("%g\t", gsl_matrix_get(m, i, j));
        }
        printf("\n");
    }
}

int main() {
    // Define matrices
    double A1_data[] = {3, -1, 1, 3, 6, 2, 3, 3, 7};
    double b1_data[] = {1, 0, 4};

    double A2_data[] = {10, -1, 0, -1, 10, -2, 0, -2, 10};
    double b2_data[] = {9, 7, 6};

    double A3_data[] = {10, 5, 0, 0, 5, 10, -4, 0, 0, -4, 8, -1, 0, 0, -1, 5};
    double b3_data[] = {6, 25, -11, -11};

    double A4_data[] = {4, 1, 1, 0, 1, -1, -3, 1, 1, 0, 2, 1, 5, -1, -1, -1, -1, -1, 4, 0, 0, 2, -1, 1, 4};
    double b4_data[] = {6, 6, 6, 6, 6};

    // Create GSL matrices and vectors
    gsl_matrix_view A1 = gsl_matrix_view_array(A1_data, 3, 3);
    gsl_vector_view b1 = gsl_vector_view_array(b1_data, 3);

    gsl_matrix_view A2 = gsl_matrix_view_array(A2_data, 3, 3);
    gsl_vector_view b2 = gsl_vector_view_array(b2_data, 3);

    gsl_matrix_view A3 = gsl_matrix_view_array(A3_data, 4, 4);
    gsl_vector_view b3 = gsl_vector_view_array(b3_data, 4);

    gsl_matrix_view A4 = gsl_matrix_view_array(A4_data, 5, 5);
    gsl_vector_view b4 = gsl_vector_view_array(b4_data, 5);

    // Perform LU decomposition
    gsl_permutation *p1 = gsl_permutation_alloc(3);
    gsl_permutation *p2 = gsl_permutation_alloc(3);
    gsl_permutation *p3 = gsl_permutation_alloc(4);
    gsl_permutation *p4 = gsl_permutation_alloc(5);

    gsl_linalg_LU_decomp(&A1.matrix, p1, NULL);
    gsl_linalg_LU_decomp(&A2.matrix, p2, NULL);
    gsl_linalg_LU_decomp(&A3.matrix, p3, NULL);
    gsl_linalg_LU_decomp(&A4.matrix, p4, NULL);

    // Solve systems
    gsl_vector *x1 = gsl_vector_alloc(3);
    gsl_vector *x2 = gsl_vector_alloc(3);
    gsl_vector *x3 = gsl_vector_alloc(4);
    gsl_vector *x4 = gsl_vector_alloc(5);

    gsl_linalg_LU_solve(&A1.matrix, p1, &b1.vector, x1);
    gsl_linalg_LU_solve(&A2.matrix, p2, &b2.vector, x2);
    gsl_linalg_LU_solve(&A3.matrix, p3, &b3.vector, x3);
    gsl_linalg_LU_solve(&A4.matrix, p4, &b4.vector, x4);

    // Print solutions
    printf("Solution for System 1:\n");
    print_matrix(x1);
    printf("\n");

    printf("Solution for System 2:\n");
    print_matrix(x2);
    printf("\n");

    printf("Solution for System 3:\n");
    print_matrix(x3);
    printf("\n");

    printf("Solution for System 4:\n");
    print_matrix(x4);
    printf("\n");

    // Free memory
    gsl_permutation_free(p1);
    gsl_permutation_free(p2);
    gsl_permutation_free(p3);
    gsl_permutation_free(p4);
    gsl_vector_free(x1);
    gsl_vector_free(x2);
    gsl_vector_free(x3);
    gsl_vector_free(x4);

    return 0;
}
