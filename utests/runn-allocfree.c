#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <time.h>

#include "runn.h"

void LogMat(float mat[], size_t rows, size_t cols)
{
	for (size_t r = 0; r < rows; r++)
	{
		printf("    ");
		for (size_t c = 0; c < cols; c++)
			printf("%f ", mat[r*cols+c]);
		printf("\n");
	}
}

bool UTRunnAllocFree()
{
	printf("[RUNN ALLOC & FREE] Testing...\n");

	bool flag = false;

	srand(1911);

	NeuralNetwork nn;

	NNLayerParams layers[] = {
		{ .size=2, .activation=ACTIVATION_TANH },
		{ .size=3, .activation=ACTIVATION_SIGMOID },
		{ .size=1, .activation=ACTIVATION_NULL }
	};

	if (!NNAlloc(&nn, 3, layers))
		return 1;

	float from = -1.0, to = 1.0;

	NNShuffle(nn, from, to, from, to);
	
	printf("  Layer 1 Weights:\n");
	LogMat(nn.layers[0].weights, 3, 2);
	printf("  Layer 1 Biases:\n");
	LogMat(nn.layers[0].biases, 3, 1);
	printf("  Layer 2 Weights:\n");
	LogMat(nn.layers[1].weights, 1, 3);
	printf("  Layer 2 Biases:\n");
	LogMat(nn.layers[1].biases, 1, 1);

	NNFree(&nn);

	flag = true;

	printf("[RUNN ALLOC & FREE] -> %s\n\n", (flag ? "YAS<3" : "NAH:("));

	return flag;
}
