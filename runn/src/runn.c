// Rami's Usable Neural Network

#include <stddef.h>
#include <stdbool.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include "runn.h"

/* Neural Network */

/*
 *  NN->Net->Activation
 */

// Linear
float ActivLinear(float val)
{
	return val;
}

float ActivLinearDeriv(float val)
{
	return 1.0;
}

// Sigmoid
float ActivSigmoid(float val)
{
	return 1.0 / (1.0 + exp(-val));
}

float ActivSigmoidDeriv(float val)
{
	float sigmoid = ActivSigmoid(val);
	return sigmoid * (1.0 - sigmoid);
}

// Tanh
float ActivTanh(float val)
{
	return tanh(val);
}

float ActivTanhDeriv(float val)
{
	double _tanh = tanh(val);
	return 1.0 - _tanh * _tanh;
}

// ReLU
float ActivReLU(float val)
{
	return (val >= 0.0 ? val : 0.0);
}

float ActivReLUDeriv(float val)
{
	return (val >= 0.0 ? 1.0 : 0.0);
}

/*
 *  NN->Net->Loss
 */

float LossMSE(size_t size, float act[], float exp[])
{
	float sum = 0;
	for (size_t i = 0; i < size; i++)
		sum += pow(act[i]-exp[i], 2);
	return sum/size;
}

void LossMSEDeriv(size_t size, float act[], float exp[], float out[])
{
	for (size_t i = 0; i < size; i++)
		out[i] = 2.0*(act[i]-exp[i])/size;
}

/*
 *  NN->Net->Layer
 */ 

bool NetLayerAlloc(NetLayer *layer, size_t size, size_t nextSize, NetActiv activ)
{
	*layer = (NetLayer){
		.size     = size,
		.nextSize = nextSize,
		.weights  = NULL,
		.biases   = NULL,
		.denseIn  = calloc(size,     sizeof(*layer->denseIn)),
		.activIn  = calloc(nextSize, sizeof(*layer->activIn)),
		.activ    = activ };

	if (nextSize != 0)
	{
		layer->weights = calloc(nextSize*size, sizeof(*layer->weights));
		layer->biases  = calloc(nextSize,      sizeof(*layer->biases));
		
		return
			   layer->weights
			&& layer->biases
			&& layer->denseIn
			&& layer->activIn;
	}
	
	return true;
}

void NetLayerFree(NetLayer *layer)
{
	free(layer->weights);
	free(layer->biases);
	free(layer->denseIn);
	free(layer->activIn);
}

// Allows to in == out
bool NetLayerForward(NetLayer *layer, float *in, float *out)
{
	// Setting dense layer X
	for (int r = 0; r < layer->size; r++)
		layer->denseIn[r] = in[r];

	// Y = func(W dot X + B)
	for (int r = 0; r < layer->nextSize; r++)
	{
		// Setting activ layer X
		layer->activIn[r] = layer->biases[r];
		for (int c = 0; c < layer->size; c++)
			layer->activIn[r] += layer->weights[r*layer->size+c] * layer->denseIn[c];
		// Setting out
		out[r] = layer->activ.func(layer->activIn[r]);
	}
}

// Does NOT allows to in == out
bool NetLayerBackwardGD(NetLayer *layer, float gradOut[], float gradIn[], float lrate)
{	
	// Activation layer
	// Xa - activ layer in 
	// dE/dXa = dE/dY * f'(Xa)
	float *gradDenseOut = calloc(layer->nextSize, sizeof(*gradDenseOut));
	for (size_t i = 0; i < layer->nextSize; i++)
		gradDenseOut[i] = gradOut[i] * layer->activ.deriv(layer->activIn[i]);

	// Dense layer
	// X - dense layer in 
	// dE/dW = dE/dY * X^T 
	// W = W' - lrate * dE/dW
	// B = B' - lrate * dE/dW
	for (int i = 0; i < layer->nextSize; i++)
	{
		for (int j = 0; j < layer->size; j++)
			layer->weights[i*layer->size+j]
				-= lrate * gradDenseOut[i] * layer->denseIn[j];
		layer->biases[i] -= lrate * gradDenseOut[i];
	}

	// Return gradIn
	for (int i = 0; i < layer->size; i++)
	{
		gradIn[i] = 0.0;
		for (int j = 0; j < layer->nextSize; j++)
			gradIn[i] += layer->weights[j*layer->nextSize+i] * gradDenseOut[j];
	}
	free(gradDenseOut);
}


bool NetAlloc(
	Net *net,
	size_t lcount,
	size_t lsizes[],
	NetActiv activs[])
{
	*net = (Net){
		.lcount = lcount,
		.layers = malloc(lcount*sizeof(*net->layers))
	};

	if (!net->layers)
		return false;

	for (size_t i = 0; i < net->lcount; i++)
	{
		if (!NetLayerAlloc(
			net->layers+i,
			lsizes[i],
			(i != net->lcount-1 ? lsizes[i+1] : 0),
			(i != net->lcount-1 ? activs[i]   : ACTIV_NULL)))
		{
			for (int j = 0; j < i; j++)
				NetLayerFree(net->layers+i);

			free(net->layers);

			return false;
		}
	}

	return true;
}

void NetFree(Net *net)
{
	if (net->layers == NULL)
		return;

	for (size_t i = 0; i < net->lcount; i++)
		NetLayerFree(net->layers+i);

	free(net->layers);
}

void NetForward(Net *net, float in[], float out[])
{
	size_t maxLayerSize = 0;
	for (size_t l = 0; l < net->lcount; l++)
		if (net->layers[l].size > maxLayerSize)
			maxLayerSize = net->layers[l].size;

	// Create the overshoot matrix
	float *buffer = malloc(maxLayerSize*sizeof(*buffer));

	for (size_t i = 0; i < net->layers[0].size; i++)
		buffer[i] = in[i];

	// Feed forward
	for (size_t l = 0; l < net->lcount-1; l++)
		NetLayerForward(&net->layers[l], buffer, buffer);

	for (size_t i = 0; i < net->layers[net->lcount-1].size; i++)
		out[i] = buffer[i];
	
	free(buffer);
}

void ArrayRandomize(float *arr, float from, float to, size_t size)
{
	for (size_t i = 0; i < size; i++)
		arr[i] = from+(float)rand()/RAND_MAX*(from-to);
}

void NetShuffle(Net *net)
{	
	for (size_t l = 0; l < net->lcount-1; l++)
	{
		ArrayRandomize(
			net->layers[l].weights,
			1.0,
			0.0,
			net->layers[l].nextSize*net->layers[l].size);

		ArrayRandomize(
			net->layers[l].biases,
			1.0,
			0.0,
			net->layers[l].nextSize);
	}
}
