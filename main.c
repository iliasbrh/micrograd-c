#include <stdio.h>
#include <stdlib.h>
#include "arena.h"
#include "value.h"
#include "matrix.h"
#include "read_ubyte.h"
#include "optimizer.h"

int main(void) {
		mem_arena* perm_arena = create_arena(GiB(3));

		// data loading
		u64 train_sample_length = 50000; // <= 50000
		u64 test_sample_length = 10000; // <= 10000
		
		FILE* train_img_file = fopen("mnist/train-images.idx3-ubyte", "rb");
		img_dataset train_img = load_img_dataset(perm_arena, train_img_file, train_sample_length);
		
		FILE* train_lbl_file = fopen("mnist/train-labels.idx1-ubyte", "rb");
		lbl_dataset train_lbl = load_lbl_dataset(perm_arena, train_lbl_file, train_sample_length);


		FILE* test_img_file = fopen("mnist/test-images.idx3-ubyte", "rb");
		img_dataset test_img = load_img_dataset(perm_arena, test_img_file, test_sample_length);
		
		FILE* test_lbl_file = fopen("mnist/test-labels.idx1-ubyte", "rb");
		lbl_dataset test_lbl = load_lbl_dataset(perm_arena, test_lbl_file, test_sample_length);

		// weight init
		f32 range = sqrtf(6.0f)/sqrtf((f32)(16+784));
		matrix* weight1 = random_fill(perm_arena, 16, 784, -range, range);
		matrix* bias1 = zeros(perm_arena, 16, 1);
		range = sqrtf(6.0f)/sqrtf((f32)(16+16));
		matrix* weight2 = random_fill(perm_arena, 16, 16, -range, range);
		matrix* bias2 = zeros(perm_arena, 16, 1);
		range = sqrtf(6.0f)/sqrtf((f32)(16+10));
		matrix* weight3 = random_fill(perm_arena, 10, 16, -range, range);
		matrix* bias3 = zeros(perm_arena, 10, 1);
		
		u64 pos_after_params = perm_arena->pos;	

		u64 epochs = 100;
		u32 batch_size = 16;

		adam_optimizer adam;
		adam.lr = 0.001f / (f32)batch_size;
		adam.beta1 = 0.9f;
		adam.beta2 = 0.999f;
		adam.beta1_t = adam.beta1;
		adam.beta2_t = adam.beta2;

		
		for (u64 epoch=0; epoch<epochs; ++epoch) {
				f32 running_loss = 0.0f;

				// training_loop
				for (u64 i=0; i<train_sample_length; ++i) {

						// zero out the grads of the dataset
						zero_grad(train_img[i]);

						// forward pass
						matrix* x1 = mat_multiply(perm_arena, weight1, train_img[i]);
						matrix* x1_b = mat_add(perm_arena, bias1, x1);
						matrix* x1_rel = mat_relu(perm_arena, x1_b);

						matrix* x2 = mat_multiply(perm_arena, weight2, x1_rel);
						matrix* x2_b = mat_add(perm_arena, bias2, x2);
						matrix* x2_rel = mat_relu(perm_arena, x2_b);

						matrix* x3 = mat_multiply(perm_arena, weight3, x2_rel);
						matrix* x3_b = mat_add(perm_arena, bias3, x3);

						matrix* lbl = one_hot(perm_arena, 10, train_lbl[i]);

						value* loss = mse_loss(perm_arena, x3_b, lbl);
						running_loss += loss->data;
						
						// back propagation
						backward(loss);

						arena_pop_to(perm_arena, pos_after_params);
						
						if ((i+1)%batch_size == 0) {
								// step
								matrix_grad_step(weight1, adam);
								matrix_grad_step(weight2, adam);
								matrix_grad_step(weight3, adam);
								matrix_grad_step(bias1, adam);
								matrix_grad_step(bias2, adam);
								matrix_grad_step(bias3, adam);

								zero_grad(weight1);
								zero_grad(weight2);
								zero_grad(weight3);
								zero_grad(bias1);
								zero_grad(bias2);
								zero_grad(bias3);

								adam.beta1_t *= adam.beta1;
								adam.beta2_t *= adam.beta2;
						}
				}
				printf("Loss after %lu epochs : %f\n", epoch+1, running_loss/train_sample_length);

				// testing loop
				u64 correct = 0;
				for (u64 i=0; i<test_sample_length; ++i) {
						matrix* x1 = mat_multiply(perm_arena, weight1, test_img[i]);
						matrix* x1_b = mat_add(perm_arena, bias1, x1);
						matrix* x1_rel = mat_relu(perm_arena, x1_b);

						matrix* x2 = mat_multiply(perm_arena, weight2, x1_rel);
						matrix* x2_b = mat_add(perm_arena, bias2, x2);
						matrix* x2_rel = mat_relu(perm_arena, x2_b);

						matrix* x3 = mat_multiply(perm_arena, weight3, x2_rel);
						matrix* x3_b = mat_add(perm_arena, bias3, x3);

						u8 argmax = 0;
						for (u8 i=1; i<10; ++i)
								if(x3_b->values[i]->data > x3_b->values[argmax]->data) argmax = i;

						if (argmax == test_lbl[i]) correct++;

						arena_pop_to(perm_arena, pos_after_params);
				}
				f32 accuracy = (f32)correct / (f32)test_sample_length;
				printf("Accuracy : %.2f%% \n", accuracy*100.);
		}

		fclose(train_img_file);
		fclose(train_lbl_file);
		fclose(test_img_file);
		fclose(test_lbl_file);


		destroy_arena(perm_arena);

		return 0;
}
