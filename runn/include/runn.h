#ifndef RUNN_H
#define RUNN_H

#include <stddef.h>
#include <stdbool.h>

/* Neural Network */

// ------------------------
// NN->Activation Functions
// ------------------------

typedef struct NNActivation
{
	float (*func)(float);
	float (*deriv)(float);
} NNActivation;

// NULL
#define ACTIVATION_NULL (NNActivation){NULL,NULL}

// Linear
float ActivationLinear(float val);
float ActivationLinearDeriv(float val);
#define ACTIVATION_LINEAR (NNActivation){ActivationLinear,ActivationLinearDeriv}

// Sigmoid
float ActivationSigmoid(float val);
float ActivationSigmoidDeriv(float val);
#define ACTIVATION_SIGMOID (NNActivation){ActivationSigmoid,ActivationSigmoidDeriv}

// Tanh 
float ActivationTanh(float val);
float ActivationTanhDeriv(float val);
#define ACTIVATION_TANH (NNActivation){ActivationTanh,ActivationTanhDeriv}

// ReLU 
float ActivationReLU(float val);
float ActivationReLUDeriv(float val);
#define ACTIVATION_RELU (NNActivation){ActivationReLU,ActivationReLUDeriv}

// ------------------
// NN->Loss Functions
// ------------------

float LossMSE(size_t size, float act[], float exp[]);
void LossMSEDeriv(size_t size, float act[], float exp[], float out[]);

// ------------------------------------
// NN->Layer->Initialization Parameters
// ------------------------------------

typedef struct NNLayerParams
{
	size_t size;
	NNActivation activ;
} NNLayerParams;

// ---------
// NN->Layer
// ---------

typedef struct NNLayer
{
	size_t size;
	size_t nsize;
	float *weights; // next x size
	float *biases;  // next x 1
	float *denseIn; // size x 1
	float *activIn; // next x 1
	NNActivation activ;
} NNLayer;

// WARNING: Do not alloc twice
bool NNLayerAlloc(NNLayer *layer, NNLayerParams params, size_t nsize);

void NNLayerFree(NNLayer *layer);

// Can be in == out
bool NNLayerForward(NNLayer *layer, float in[], float out[]);

// Can NOT be in == out
bool NNLayerBackwardGD(NNLayer *layer, float gradOut[], float gradIn[], float lrate);

// --------------
// Neural Network
// --------------

typedef struct NeuralNetwork
{
	size_t    lcount;
	NNLayer *layers;
} NeuralNetwork;

bool NNAlloc(
	NeuralNetwork *nn,
	size_t lcount,
	NNLayerParams lparams[]);

void NNFree(NeuralNetwork *nn);

void NNForward(NeuralNetwork *nn, float in[], float out[]);

void NNShuffle(NeuralNetwork *nn);

#endif /* RUNN_H */
