// Rami's Usable Neural Network

#include <stddef.h>
#include <stdbool.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include "runn.h"

bool MatAlloc(Mat *mat, size_t rows, size_t cols)
{
	*mat = (Mat){
		.rows=rows,
		.cols=cols,
		.data=calloc(rows*cols, sizeof(*mat->data))
	};

	return mat->data!=NULL;
}

void MatFree(Mat *mat)
{
	free(mat->data);
}

void MatLog(Mat *mat)
{
	printf("%dx%d\n", mat->rows, mat->cols);
	for (int r = 0; r < mat->rows; r++)
	{
		for (int c = 0; c < mat->cols; c++)
			printf("%lf ", MATAT(*mat,r,c));
		printf("\n");
	}
}

void MatRandomize(Mat *mat, NN_DATA_TYPE from, NN_DATA_TYPE to)
{
	for (int r = 0; r < mat->rows; r++)
		for (int c = 0; c < mat->cols; c++)
			MATAT(*mat,r,c) = from+(NN_DATA_TYPE)rand()/RAND_MAX*(to-from);
}

void MatAssign(Mat *mat, NN_DATA_TYPE data[])
{
	for (int i = 0; i < mat->rows*mat->cols; i++)
		mat->data[i] = data[i];
}

// NET

// ACTIV

// Linear
NN_DATA_TYPE ActivLinear(NN_DATA_TYPE val)
{
	return val;
}

NN_DATA_TYPE ActivLinearDeriv(NN_DATA_TYPE val)
{
	return 1.0;
}

// Sigmoid
NN_DATA_TYPE ActivSigmoid(NN_DATA_TYPE val)
{
	return 1.0 / (1.0 + exp(-val));
}

NN_DATA_TYPE ActivSigmoidDeriv(NN_DATA_TYPE val)
{
	NN_DATA_TYPE sigmoid = ActivSigmoid(val);
	return sigmoid * (1.0 - sigmoid);
}

// Tanh
NN_DATA_TYPE ActivTanh(NN_DATA_TYPE val)
{
	return tanh(val);
}

NN_DATA_TYPE ActivTanhDeriv(NN_DATA_TYPE val)
{
	double _tanh = tanh(val);
	return 1.0 - _tanh * _tanh;
}

// ReLU
NN_DATA_TYPE ActivReLU(NN_DATA_TYPE val)
{
	return (val >= 0.0 ? val : 0.0);
}

NN_DATA_TYPE ActivReLUDeriv(NN_DATA_TYPE val)
{
	return (val >= 0.0 ? 1.0 : 0.0);
}

// LAYER 
bool NetLayerAlloc(NetLayer *layer, size_t size, size_t nextSize, NetActiv activ)
{
	*layer = (NetLayer){ .size=size, .activ=activ };
	return
		   MatAlloc(&layer->weights, nextSize, size)
		&& MatAlloc(&layer->biases,  nextSize, 1)
		&& MatAlloc(&layer->denseIn, size, 1)
		&& MatAlloc(&layer->activIn, nextSize, 1);
}

void NetLayerFree(NetLayer *layer)
{
	MatFree(&layer->weights);
	MatFree(&layer->biases);
	MatFree(&layer->denseIn);
	MatFree(&layer->activIn);
}

bool NetLayerForward(NetLayer *layer, NN_DATA_TYPE in[], NN_DATA_TYPE out[])
{
	Mat matIn, matOut;
	if (!MatAlloc(&matIn, layer->size, 1) || !MatAlloc(&matOut, layer->nextSize, 1))
	{
		MatFree(&matIn);
		MatFree(&matOut);
		return false;
	}

	MatAssign(matIn);
	
	MatLog(&matIn);
	MatLog(&matOut);

	MatFree(&matIn);
	MatFree(&matOut);
}
