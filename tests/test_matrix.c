#include "arena.h"
#include "value.h"
#include "matrix.h"
#include "types.h"
#include <stdio.h>

int main(void) {
		mem_arena* perm_arena = create_arena(MiB(10));

		printf("%i", perm_arena->pos);
		printf("\n");
		matrix* mat1 = zeros(perm_arena, 3, 3);
		
		mat1->values[0]->data += 1.;
		mat1->values[1]->data += 1.;
		mat1->values[2]->data += 1.;
		mat1->values[3]->data += 1.;
		mat1->values[4]->data += 1.;
		mat1->values[5]->data += 1.;
		mat1->values[6]->data += 1.;
		mat1->values[7]->data += 1.;
		mat1->values[8]->data += 1.;
		print_matrix(mat1);
		printf("%i", perm_arena->pos);
		printf("\n");

		
		matrix* mat2 = zeros(perm_arena, 3, 3);
		
		mat2->values[0]->data += 1.;
		mat2->values[4]->data += 2.;
		mat2->values[8]->data += 1.;
		print_matrix(mat2);
		printf("%i", perm_arena->pos);
		printf("\n");

		
		matrix* out = mat_multiply(perm_arena, mat1, mat2);
		print_matrix(out);
		printf("%i", perm_arena->pos);
		printf("\n");

		matrix* negative = mat_neg(perm_arena, out);
		print_matrix(negative);
		printf("%i", perm_arena->pos);
		printf("\n");

		matrix* onepointfive = zeros(perm_arena, 3, 3);
		printf("%i", perm_arena->pos);
		printf("\n");
		
		onepointfive->values[0]->data += 1.5;
		onepointfive->values[1]->data += 1.5;
		onepointfive->values[2]->data += 1.5;
		onepointfive->values[3]->data += 1.5;
		onepointfive->values[4]->data += 1.5;
		onepointfive->values[5]->data += 1.5;
		onepointfive->values[6]->data += 1.5;
		onepointfive->values[7]->data += 1.5;
		onepointfive->values[8]->data += 1.5;

		matrix* separated = mat_add(perm_arena, negative, onepointfive);
		print_matrix(separated);
		printf("%i", perm_arena->pos);
		printf("\n");
		matrix* rel = mat_relu(perm_arena, separated);
		print_matrix(rel);
		printf("%i", perm_arena->pos);
		printf("\n");
		matrix* exp = mat_exp(perm_arena, rel);
		print_matrix(exp);
		printf("%i", perm_arena->pos);
		printf("\n");

		matrix* ones_column = ones(perm_arena, 3, 1);
		matrix* zeros_column = zeros(perm_arena, 3, 1);
		print_matrix(ones_column);
		print_matrix(zeros_column);
		value* mse = mse_loss(perm_arena, ones_column, zeros_column);
		printf("%.4f", mse->data);

		printf("\n");
		printf("%i", perm_arena->pos);
		printf("\n");


		printf("%i", perm_arena->pos);

		destroy_arena(perm_arena);

		return 0;
}
