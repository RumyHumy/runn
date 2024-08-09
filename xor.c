#import <stdio.h>
#import <time.h>

#import "runn.h"

int main()
{
	srand(time(NULL));

	Net net;

	size_t lsizes[] = { 2, 3, 1 };

	NetActiv activs[] = {
		ACTIV_TANH,
		ACTIV_SIGMOID
	};

	if (!NetAlloc(&net, 3, lsizes, activs))
		return 1;

	NetShuffle(&net);

	float eIn[4][2] = { {0, 0},
	                    {0, 1},
	                    {1, 0},
                        {1, 1}};

	float eOut[4][1] = { { 0 }, { 1 }, {1}, {0} };

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
		NetForward(&net, eIn[0], out);
		printf("%f xor %f = %f ~ %f\n", eIn[i][0], eIn[i][1], out[0], eOut[i][0]);
	}

	return 0;
}
