#include <stdio.h>
#include <stdlib.h>
#include "arena.h"
#include "value.h"
#include "matrix.h"
#include "read_ubyte.h"

int main(void) {
		mem_arena* arena = create_arena(MiB(1));

		FILE* inputFile = fopen("test-images.idx3-ubyte", "rb");
		u8 buffer;
		for (u8 i=0; i<16; ++i) fread(&buffer, 1, 1, inputFile); // clearing header data
		matrix* out = read_ubyte_image(arena, inputFile, 28, 28);

		print_matrix(out);

		FILE* inputLabels = fopen("test-labels.idx1-ubyte", "rb");
		for (u8 i=0; i<8; ++i) fread(&buffer, 1, 1, inputLabels); // clearing header data
		value* label = read_ubyte_label(arena, inputLabels);
		printf("%f", label->data);

		destroy_arena(arena);
		fclose(inputFile);
}
