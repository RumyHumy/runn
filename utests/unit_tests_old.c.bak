#include <stddef.h>
#include <stdbool.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>

#include "runn.h" 

#define PF(flag) (flag ? "YAS<3" : "NAH:(")
#define OE(flag) (flag ? "OK"   : "ERROR")

bool Compare(float val1, float val2, int p)
{
	int prec = pow(10,p);
	return roundf(val1*prec)/prec == roundf(val2*prec)/prec;
}

void ArrayCopy(float *src, float *dst, size_t size)
{
	for (size_t i = 0; i < size; i++)
		dst[i] = src[i];
}

bool ArrayCompare(float *arr1, float *arr2, size_t size, int precision)
{
	bool flag = true;
	for (int i = 0; i < size; i++) {
		//printf("%f ~ %f\n", arr1[i], arr2[i]);
		flag &= Compare(arr1[i], arr2[i], precision);
	}
	return flag;
}

bool UTNetActiv()
{
	bool flag_linear =
		   Compare(ACTIV_LINEAR.func(+0.900),  +0.900, 3)
		&& Compare(ACTIV_LINEAR.func(-3.200),  -3.200, 3)
		&& Compare(ACTIV_LINEAR.deriv(+0.200), +1.000, 3)
		&& Compare(ACTIV_LINEAR.deriv(-2.400), +1.000, 3);
	printf("  ActivLinear  -> %s\n", OE(flag_linear));

	bool flag_sigmoid =
		   Compare(ACTIV_SIGMOID.func(+0.900),  +0.711, 3)
		&& Compare(ACTIV_SIGMOID.func(-3.200),  +0.039, 3)
		&& Compare(ACTIV_SIGMOID.deriv(+0.200), +0.248, 3)
		&& Compare(ACTIV_SIGMOID.deriv(-2.400), +0.076, 3);
	printf("  ActivSigmoid -> %s\n", OE(flag_sigmoid));

	bool flag_tanh =
		   Compare(ACTIV_TANH.func(+0.900),  +0.716, 3)
		&& Compare(ACTIV_TANH.func(-3.200),  -0.997, 3)
		&& Compare(ACTIV_TANH.deriv(+0.200), +0.961, 3)
		&& Compare(ACTIV_TANH.deriv(-2.400), +0.032, 3);
	printf("  ActivTanh    -> %s\n", OE(flag_tanh));

	bool flag_relu =
		   Compare(ACTIV_RELU.func(+0.900),  +0.900, 3)
		&& Compare(ACTIV_RELU.func(-3.200),  +0.000, 3)
		&& Compare(ACTIV_RELU.deriv(+0.200), +1.000, 3)
		&& Compare(ACTIV_RELU.deriv(-2.400), +0.000, 3);
	printf("  ActivRelu    -> %s\n", OE(flag_relu));

	return
		   flag_linear
		&& flag_sigmoid
		&& flag_tanh
		&& flag_relu;
}

bool UTNetLayerAllocFree()
{
	bool flag = false;

	NetLayer layer1, layer2, layer3;
	if (
		   !NetLayerAlloc(&layer1, (NetLayerParams){3, ACTIV_LINEAR}, 4)
		|| !NetLayerAlloc(&layer2, (NetLayerParams){2, ACTIV_SIGMOID}, 3)
		|| !NetLayerAlloc(&layer3, (NetLayerParams){5, ACTIV_RELU}, 5))
		goto cleanup;

	flag =
		   layer1.activ.func   == ActivLinear 
		&& layer1.activ.deriv  == ActivLinearDeriv
		&& layer2.activ.func   == ActivSigmoid 
		&& layer2.activ.deriv  == ActivSigmoidDeriv
		&& layer3.activ.func   == ActivReLU
		&& layer3.activ.deriv  == ActivReLUDeriv;

cleanup:
	NetLayerFree(&layer1);
	NetLayerFree(&layer2);
	NetLayerFree(&layer3);

	return flag;
}

#define LFORWARD(layern, flagn) \
	ArrayCopy(ws, layern.weights, layern.size*layern.nextSize); \
	ArrayCopy(bs, layern.biases,  layern.nextSize); \
	NetLayerForward(&layern, in, out); \
	flagn &= ArrayCompare(out, eOut, layern.nextSize, 3);

bool UTNetLayerForward()
{
	NetLayer layer1, layer2, layer3;
	if (
		   !NetLayerAlloc(&layer1, (NetLayerParams){1, ACTIV_LINEAR}, 1)
		|| !NetLayerAlloc(&layer2, (NetLayerParams){1, ACTIV_SIGMOID}, 2)
		|| !NetLayerAlloc(&layer3, (NetLayerParams){3, ACTIV_TANH}, 2))
		goto cleanup;

	bool flag1 = true;
	bool flag2 = true;
	bool flag3 = true;

	// LAYER 1
	// Test 1
	{
		float ws[1]   = { +2.0 };
		float bs[1]   = { -0.5 };
		float in[1]   = { +0.5 };
		float eOut[1] = { +0.5 };
		float out[1]  = {  0.0 };
		LFORWARD(layer1, flag1)
	}
	// Test 2
	{
		float ws[1*1] = { +2.0 };
		float bs[1]   = { -0.5 };
		float in[1]   = { -0.5 };
		float eOut[1] = { -1.5 };
		float out[1];
		LFORWARD(layer1, flag1)
	}
	printf("  TEST LAYER 1 -> %s\n", OE(flag1));

	// LAYER 2
	// Test 1
	{
		float ws[2*1] = { +2.0,
		                  +0.5 };
		float bs[2]   = { -0.5,
		                  +1.0 };
		float in[1]   = { +2.0 };
		float eOut[2] = { +0.971,
		                  +0.881 };
		float out[2];
		LFORWARD(layer2, flag2)
	}
	// Test 2
	{
		float ws[2*1] = { +2.0,
		                  +0.5 };
		float bs[2]   = { -0.5,
		                  +1.0 };
		float in[1]   = { -1.0 };
		float eOut[2] = { 0.076,
		                  0.622 };
		float out[2];
		LFORWARD(layer2, flag2)
	}
	printf("  TEST LAYER 2 -> %s\n", OE(flag2));

	// LAYER 2
	// Test 1
	{
		float ws[2*3] = { +0.5, -1.0,  0.0,
		                   0.0, +2.0, -0.5 };
		float bs[2]   = { +0.5,
		                  +1.0 };
		float in[3]   = { +2.0,
		                  -1.0,
		                  +4.0 };
		float eOut[2] = { +0.987,
		                  -0.995 };
		float out[2];
		LFORWARD(layer3, flag3)
	}
	printf("  TEST LAYER 3 -> %s\n", OE(flag3));



cleanup:
	NetLayerFree(&layer1);
	NetLayerFree(&layer2);
	NetLayerFree(&layer3);

	return flag1 && flag2 && flag3;
}

#define LBACKWARDGD(layern, flagn) \
	for (int i = 0; i < 10000; i++) \
	{ \
		NetLayerForward(&layern, in, out); \
		LossMSEDeriv(layern.nextSize, out, eOut, grad); \
		NetLayerBackwardGD(&layern, grad, gradIn, 0.05); \
	} \
	NetLayerForward(&layern, in, out); \
	flagn &= ArrayCompare(out, eOut, layern.nextSize, 3);

bool UTNetLayerBackwardGD()
{
	NetLayer layer1, layer2;
	if (
		   !NetLayerAlloc(&layer1, (NetLayerParams){1, ACTIV_LINEAR}, 1)
		|| !NetLayerAlloc(&layer2, (NetLayerParams){2, ACTIV_TANH}, 3))
		goto cleanup;

	bool flag1 = true;
	bool flag2 = true;

	// LAYER 1
	// Test 1
	{
		float eWs[1]  = { +2.0 };
		float eBs[1]  = { -0.5 };
		float in[1]   = { +0.5 };
		float eOut[1] = { +0.5 };
		float out[1];
		float grad[1];
		float gradIn[1];
		LBACKWARDGD(layer1, flag1)
	}
	// Test 2
	{
		float eWs[1]  = { +6.0 };
		float eBs[1]  = { -4.5 };
		float in[1]   = { -1.0 };
		float eOut[1] = { +0.77 };
		float out[1];
		float grad[1];
		float gradIn[1];
		LBACKWARDGD(layer1, flag1)
	}
	printf("  TEST LAYER 1 -> %s\n", OE(flag1));

	// LAYER 2
	// Test 1
	{
		float eWs[3*2] = { +2.0, +1.0,
		                   -0.5, -1.5,
		                   +0.1, -2.5 };
		float eBs[3]   = { -0.5,
		                   -1.5,
		                   +5.0 };
		float in[2]    = { +0.5,
		                   -1.0 };
		float eOut[3]  = { -0.77,
		                   +0.52 };
		float out[3];
		float grad[3];
		float gradIn[2];
		LBACKWARDGD(layer2, flag2)
	}
	// Test 2
	{
		float eWs[3*2] = { +1.0, +6.0,
		                   -1.5, -1.5,
		                   +9.1, -2.5 };
		float eBs[3]   = { -0.5,
		                   -1.5,
		                   +5.0 };
		float in[2]    = { +0.0,
		                   +6.0 };
		float eOut[3]  = { +0.19,
		                   -0.11};
		float out[3];
		float grad[3];
		float gradIn[2];
		LBACKWARDGD(layer2, flag2)
	}
	printf("  TEST LAYER 2 -> %s\n", OE(flag2));

cleanup:
	NetLayerFree(&layer1);
	NetLayerFree(&layer2);

	return flag1 && flag2;
}

bool UTNetAllocFree()
{
	bool flag = false;

	Net net;
	NetLayerParams lparams[] = {
		{ 2, ACTIV_TANH },
		{ 3, ACTIV_SIGMOID },
		{ 1, ACTIV_NULL }
	};
	if (!NetAlloc(&net, 3, lparams))
		goto cleanup;

	flag =
		   net.layers[0].size  == 2
		&& net.layers[0].activ.func  == ActivTanh
		&& net.layers[0].activ.deriv == ActivTanhDeriv
		&& net.layers[1].size  == 3
		&& net.layers[1].activ.func  == ActivSigmoid
		&& net.layers[1].activ.deriv == ActivSigmoidDeriv
		&& net.layers[2].size  == 1
		&& net.layers[2].activ.func  == NULL
		&& net.layers[2].activ.deriv == NULL;

cleanup:
	NetFree(&net);

	return flag;
}


bool UTNetTrainXOR()
{
	bool flag = false;

	Net net;
	NetLayerParams lparams[] = {
		{ 2, ACTIV_TANH },
		{ 3, ACTIV_SIGMOID },
		{ 1, ACTIV_NULL }
	};
	if (!NetAlloc(&net, 3, lparams))
		goto cleanup;

	NetShuffle(&net);

	float eIn[4][2] = { {0, 0},
	                    {0, 1},
	                    {1, 0},
                        {1, 1}};
	float eOut[4][1] = { {0}, {1}, {1}, {0} };

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

	flag = true;

	NetForward(&net, eIn[0], out);
	printf("0 xor 0 = %f ~ %f\n", out[0], eOut[0][0]);
	flag &= Compare(out[0], eOut[0][0], 0);

	NetForward(&net, eIn[1], out);
	printf("0 xor 1 = %f ~ %f\n", out[0], eOut[1][0]);
	flag &= Compare(out[0], eOut[1][0], 0);

	NetForward(&net, eIn[2], out);
	printf("1 xor 0 = %f ~ %f\n", out[0], eOut[2][0]);
	flag &= Compare(out[0], eOut[2][0], 0);

	NetForward(&net, eIn[3], out);
	printf("1 xor 1 = %f ~ %f\n", out[0], eOut[3][0]);
	flag &= Compare(out[0], eOut[3][0], 0);

cleanup:
	NetFree(&net);

	return flag;
}

int main()
{
	srand(time(NULL));
	printf("[ACTIVATIONS]\n");
	printf("> %s\n\n", PF(UTNetActiv()));
	printf("[LAYER ALLOC & FREE]\n");
	printf("> %s\n\n", PF(UTNetLayerAllocFree()));
	printf("[LAYER FEED FORWARD]\n");
	printf("> %s\n\n", PF(UTNetLayerForward()));
	printf("[LAYER FEED BACKWAARD GD]\n");
	printf("> %s\n\n", PF(UTNetLayerBackwardGD()));
	printf("[NET ALLOC & FREE]\n");
	printf("> %s\n\n", PF(UTNetAllocFree()));
	printf("[NET TRAIN XOR]\n");
	printf("> %s\n\n", PF(UTNetTrainXOR()));

	return 0;
}
