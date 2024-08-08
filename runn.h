#ifndef RUNN_H
#define RUNN_H

#include <stddef.h>


/* Neural Network */

/*
 *  NN -> Net -> Activation Functions
 */

typedef struct NetActiv
{
	float (*func)(float);
	float (*deriv)(float);
} NetActiv;

// NULL
#define ACTIV_NULL (NetActiv){NULL,NULL}

// Linear
float ActivLinear(float val);
float ActivLinearDeriv(float val);
#define ACTIV_LINEAR (NetActiv){ActivLinear,ActivLinearDeriv}

// Sigmoid
float ActivSigmoid(float val);
float ActivSigmoidDeriv(float val);
#define ACTIV_SIGMOID (NetActiv){ActivSigmoid,ActivSigmoidDeriv}

// Tanh 
float ActivTanh(float val);
float ActivTanhDeriv(float val);
#define ACTIV_TANH (NetActiv){ActivTanh,ActivTanhDeriv}

// ReLU 
float ActivReLU(float val);
float ActivReLUDeriv(float val);
#define ACTIV_RELU (NetActiv){ActivReLU,ActivReLUDeriv}

/*
 *  NN -> Net -> Loss Functions
 */

float LossMSE(size_t size, float act[], float exp[]);
void LossMSEDeriv(size_t size, float act[], float exp[], float out[]);

/*
 *  NN -> Net -> Layer
 */

typedef struct NetLayer
{
	size_t size;
	size_t nextSize;
	float *weights;    // next x size
	float *biases;     // next x 1
	float *denseIn;    // size x 1
	float *activIn;    // next x 1
	NetActiv activ;
} NetLayer;

// WARNING: Do not alloc twice
bool NetLayerAlloc(NetLayer *layer, size_t size, size_t nextSize, NetActiv activ);

void NetLayerFree(NetLayer *layer);

// Can be in == out
bool NetLayerForward(NetLayer *layer, float in[], float out[]);

// Can NOT be in == out
bool NetLayerBackwardGD(NetLayer *layer, float gradOut[], float gradIn[], float lrate);

/*
 *  NN -> NET
 */

typedef struct Net
{
	size_t    lcount;
	NetLayer *layers;
} Net;

bool NetAlloc(
	Net *net,
	size_t lcount,
	size_t lsizes[],
	NetActiv activs[]);

void NetFree(Net *net);

void NetForward(Net *net, float in[], float out[]);

void NetShuffle(Net *net);

#endif /* RUNN_H */
