#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#include "types.h"

typedef struct {
		f32 lr;

		f32 beta1;
		f32 beta2;
		f32 beta1_t;
		f32 beta2_t;
} adam_optimizer;


#endif
