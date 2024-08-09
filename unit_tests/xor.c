#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "runn.h"

int main()
{
	srand(time(NULL));

	Net net;

	NetLayerParams layers[] = {
		{ .size=2, .activ=ACTIV_TANH },
		{ .size=3, .activ=ACTIV_SIGMOID },
		{ .size=1, .activ=ACTIV_NULL }
	};

	if (!NetAlloc(&net, 3, layers))
		return 1;

	NetShuffle(&net);

	float eIn[4][2]  = { { 0, 0 }, { 0, 1 }, { 1, 0 }, { 1, 1 } };
	float eOut[4][1] = { { 0    }, { 1    }, { 1    }, { 0    } };

	float out[1];
	float gradOut[3];
	float gradIn[3];

	for (int i = 0; i < 100000; i++)
	{
		int j = i%4;
		NetForward(&net, eIn[j], out);
		LossMSEDeriv(3, out, eOut[j], gradOut);

		for (int l = net.lcount-2; l >= 0; l--)
		{
			NetLayerBackwardGD(&net.layers[l], gradOut, gradIn, 0.7);
			for (int k = 0; k < 3; k++)
				gradOut[k] = gradIn[k];
		}
	}

	for (int i = 0; i < 4; i++)
	{
		NetForward(&net, eIn[i], out);
		printf("%.0f xor %.0f = %f ~ %.0f\n", eIn[i][0], eIn[i][1], out[0], eOut[i][0]);
	}

	return 0;
}
