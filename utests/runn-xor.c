#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <time.h>
#include <math.h>

#include "runn.h"

bool UTRunnXor()
{
	printf("[XOR] Testing...\n");

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

	NNShuffle(nn, -1.0, 1.0, -1.0, 1.0);

	float eIn[4][2]  = { { 0, 0 }, { 0, 1 }, { 1, 0 }, { 1, 1 } };
	float eOut[4][1] = { { 0    }, { 1    }, { 1    }, { 0    } };

	float out[1];
	float gradOut[3];
	float gradIn[3];

	for (int i = 0; i < 100000; i++)
	{
		int j = i%4;
		NNForward(nn, eIn[j], out);
		LossMSEDeriv(3, out, eOut[j], gradOut);

		for (int l = nn.lcount-2; l >= 0; l--)
		{
			NNLayerBackwardGD(nn, l, gradOut, gradIn, 0.7);
			for (int k = 0; k < 3; k++)
				gradOut[k] = gradIn[k];
		}
	}

	flag = true;
	for (int i = 0; i < 4; i++)
	{
		NNForward(nn, eIn[i], out);
		printf("  %.0f xor %.0f = %f ~ %.0f\n", eIn[i][0], eIn[i][1], out[0], eOut[i][0]);
		flag &= roundf(out[0])==eOut[i][0];
	}

	printf("[XOR] -> %s\n\n", (flag ? "YAS<3" : "NAH:("));

	NNFree(&nn);

	return flag;
}
