#include <stddef.h>
#include <stdbool.h>
#include <stdio.h>
#include <math.h>

#include "runn.h" 

#define TF(flag) (flag ? "TRUE" : "FALSE")
#define OE(flag) (flag ? "OK"   : "ERROR")

bool Compare(NN_DATA_TYPE val1, NN_DATA_TYPE val2, int p)
{
	int pot = pow(10,p);
	return roundf(val1*pot)/pot==roundf(val2*pot)/pot;
}

bool UTMatAllocFree()
{
	bool flag = false;

	Mat mat;

	if (!MatAlloc(&mat, 5, 6))
		goto cleanup;

	flag = true;
	for (int r = 0; r < mat.rows; r++)
		for (int c = 0; c < mat.cols; c++)
			flag &= (MATAT(mat,r,c) == 0.0);

cleanup:
	MatFree(&mat);

	return flag;
}

#define RAND_IN_RANGE(mat, from, to) \
	for (int r = 0; r < mat.rows; r++) \
		for (int c = 0; c < mat.cols; c++) \
			flag &= \
				   (MATAT(mat,r,c) >= from) \
				&& (MATAT(mat,r,c) <= to);

bool UTMatRandomize()
{
	bool flag = false;

	Mat mat1, mat2, mat3;
	if (
		   !MatAlloc(&mat1, 2, 3)
		|| !MatAlloc(&mat2, 5, 1)
		|| !MatAlloc(&mat3, 10, 20))
		goto cleanup;

	MatRandomize(&mat1,   0.0,  +1.0);
	MatRandomize(&mat2,  -5.0, +10.0);
	MatRandomize(&mat3, -20.0,  -5.0);

	flag = true;
	
	RAND_IN_RANGE(mat1,   0.0,  +1.0)
	RAND_IN_RANGE(mat2,  -5.0, +10.0)
	RAND_IN_RANGE(mat3, -20.0,  -5.0)

cleanup:
	MatFree(&mat1);
	MatFree(&mat2);
	MatFree(&mat3);

	return flag;
}

bool UTMatAssign()
{
	bool flag = false;

	Mat mat1, mat2, mat3;
	if (
		   !MatAlloc(&mat1, 2, 3)
		|| !MatAlloc(&mat2, 3, 2)
		|| !MatAlloc(&mat3, 2, 2))
		goto cleanup;

	MatAssign(&mat1,(NN_DATA_TYPE[]){
		+0.1, -0.2, +0.3,
		-0.4, +0.5, -0.6});

	MatAssign(&mat2,(NN_DATA_TYPE[]){
		+0.11, -0.22,
		+0.33, -0.44,
		+0.55, -0.66});

	MatAssign(&mat3,(NN_DATA_TYPE[]){
		+0.111, -0.222,
		+0.333, -0.444});

	flag =
		   Compare(MATAT(mat1,0,0), +0.100, 3)
		&& Compare(MATAT(mat1,0,1), -0.200, 3)
		&& Compare(MATAT(mat1,0,2), +0.300, 3)
		&& Compare(MATAT(mat1,1,0), -0.400, 3)
		&& Compare(MATAT(mat1,1,1), +0.500, 3)
		&& Compare(MATAT(mat1,1,2), -0.600, 3)
		&& Compare(MATAT(mat2,0,0), +0.110, 3)
		&& Compare(MATAT(mat2,0,1), -0.220, 3)
		&& Compare(MATAT(mat2,1,0), +0.330, 3)
		&& Compare(MATAT(mat2,1,1), -0.440, 3)
		&& Compare(MATAT(mat2,2,0), +0.550, 3)
		&& Compare(MATAT(mat2,2,1), -0.660, 3)
		&& Compare(MATAT(mat3,0,0), +0.111, 3)
		&& Compare(MATAT(mat3,0,1), -0.222, 3)
		&& Compare(MATAT(mat3,1,0), +0.333, 3)
		&& Compare(MATAT(mat3,1,1), -0.444, 3);

cleanup:
	MatFree(&mat1);
	MatFree(&mat2);
	MatFree(&mat3);

	return flag;
}

bool UTNetActiv()
{
	bool flag_linear =
		   Compare(ACTIV_LINEAR.func(+0.900),  +0.900, 3)
		&& Compare(ACTIV_LINEAR.func(-3.200),  -3.200, 3)
		&& Compare(ACTIV_LINEAR.deriv(+0.200), +1.000, 3)
		&& Compare(ACTIV_LINEAR.deriv(-2.400), +1.000, 3);
	printf("ActivLinear  -> %s\n", OE(flag_linear));

	bool flag_sigmoid =
		   Compare(ACTIV_SIGMOID.func(+0.900),  +0.711, 3)
		&& Compare(ACTIV_SIGMOID.func(-3.200),  +0.039, 3)
		&& Compare(ACTIV_SIGMOID.deriv(+0.200), +0.248, 3)
		&& Compare(ACTIV_SIGMOID.deriv(-2.400), +0.076, 3);
	printf("ActivSigmoid -> %s\n", OE(flag_sigmoid));

	bool flag_tanh =
		   Compare(ACTIV_TANH.func(+0.900),  +0.716, 3)
		&& Compare(ACTIV_TANH.func(-3.200),  -0.997, 3)
		&& Compare(ACTIV_TANH.deriv(+0.200), +0.961, 3)
		&& Compare(ACTIV_TANH.deriv(-2.400), +0.032, 3);
	printf("ActivTanh    -> %s\n", OE(flag_tanh));

	bool flag_relu =
		   Compare(ACTIV_RELU.func(+0.900),  +0.900, 3)
		&& Compare(ACTIV_RELU.func(-3.200),  +0.000, 3)
		&& Compare(ACTIV_RELU.deriv(+0.200), +1.000, 3)
		&& Compare(ACTIV_RELU.deriv(-2.400), +0.000, 3);
	printf("ActivRelu    -> %s\n", OE(flag_relu));

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
		   !NetLayerAlloc(&layer1, 3, 4, ACTIV_LINEAR)
		|| !NetLayerAlloc(&layer2, 2, 3, ACTIV_SIGMOID)
		|| !NetLayerAlloc(&layer3, 5, 5, ACTIV_RELU))
		goto cleanup;

	flag =
		   layer1.weights.rows == 4
		&& layer1.weights.cols == 3
		&& layer1.activ.func   == ActivLinear 
		&& layer1.activ.deriv  == ActivLinearDeriv
		&& layer2.weights.rows == 3
		&& layer2.weights.cols == 2
		&& layer2.activ.func   == ActivSigmoid 
		&& layer2.activ.deriv  == ActivSigmoidDeriv
		&& layer3.weights.rows == 5
		&& layer3.weights.cols == 5
		&& layer3.activ.func   == ActivReLU
		&& layer3.activ.deriv  == ActivReLUDeriv;

cleanup:
	NetLayerFree(&layer1);
	NetLayerFree(&layer2);
	NetLayerFree(&layer3);

	return flag;
}

bool UTNetLayerForward()
{
	NetLayer layer1, layer2, layer3;
	if (
		   !NetLayerAlloc(&layer1, 1, 1, ACTIV_SIGMOID)
		|| !NetLayerAlloc(&layer2, 2, 3, ACTIV_LINEAR)
		|| !NetLayerAlloc(&layer3, 3, 2, ACTIV_SIGMOID))
		goto cleanup;

	bool flag1 = true;
	bool flag2 = true;
	bool flag3 = true;

	{
		// LAYER 1
		NN_DATA_TYPE out[1];
		// Test 1
		// X = +0.5; W = +2.0; B = -0.50
		// Y = sigmoid(0.5 * 2.0 - 0.5) ~= 0.622
		NetLayerForward(&layer1, (NN_DATA_TYPE[]){ +0.5 }, out);
		flag1 &= Compare(out[0], 0.622, 3);
		// Test 2
		// X = -0.5; W = +1.0; B = -0.50
		// Y = sigmoid((-0.5) * 1.0 - 0.5) ~= 0.269
		NetLayerForward(&layer1, (NN_DATA_TYPE[]){ -0.5 }, out);
		flag1 &= Compare(out[0], 0.269, 3);
	}


cleanup:
	NetLayerFree(&layer1);
	NetLayerFree(&layer2);
	NetLayerFree(&layer3);

	return flag1 && flag2 && flag3;
}

int main()
{
	printf("MAT ALLOC & FREE...\n");
	printf("MAT ALLOC & FREE --------------> %s\n", TF(UTMatAllocFree()));
	printf("MAT RANDOMIZE...\n");
	printf("MAT RANDOMIZE -----------------> %s\n", TF(UTMatRandomize()));
	printf("MAT ASSIGN...\n");
	printf("MAT ASSIGN --------------------> %s\n", TF(UTMatAssign()));
	printf("NET ACTIVATIONS...\n");
	printf("NET ACTIVATIONS ---------------> %s\n", TF(UTNetActiv()));
	printf("NET LAYER ALLOC & FREE...\n");
	printf("NET LAYER ALLOC & FREE --------> %s\n", TF(UTNetLayerAllocFree()));
	printf("NET LAYER FEED FORWARD...\n");
	printf("NET LAYER FEED FORWARD --------> %s\n", TF(UTNetLayerForward()));
	return 0;
}
